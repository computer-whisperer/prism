//! Config file modification watcher.
//!
//! Port of `niri/src/utils/watcher.rs`. A dedicated thread polls the config
//! file every 500ms — polling (vs inotify) deliberately survives the hostile
//! cases: editors that rename-replace, symlinked configs whose target swaps
//! without an mtime change (nix-style symlink farms keep mtime at the epoch),
//! and the file or its parent directory not existing yet. Each poll compares
//! mtime *and* the canonicalized path; include files are re-stat'd too.
//!
//! On change the watcher re-parses on its own thread (parsing is
//! milliseconds, but no reason to block the compositor) and sends the result
//! over a calloop channel; the main loop applies it via
//! `PrismState::reload_config`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, SystemTime};
use std::{io, thread};

use calloop::channel::SyncSender;
use prism_config::{Config, ConfigParseResult, ConfigPath};

const POLLING_INTERVAL: Duration = Duration::from_millis(500);

/// Handle to the watcher thread. Dropping it disconnects the trigger
/// channel, which makes the thread exit at its next poll tick.
pub struct Watcher {
    load_config: mpsc::Sender<Option<String>>,
}

struct WatcherInner {
    /// The paths we're watching.
    path: ConfigPath,

    /// Last observed props of the watched file.
    last_props: Option<Props>,

    /// Last observed props for included files.
    includes: HashMap<PathBuf, Option<Props>>,
}

/// Properties of the watched file.
///
/// Equality on this means the file did not change.
#[derive(Debug, PartialEq, Eq)]
struct Props {
    /// Modification time of the watched file.
    mtime: SystemTime,

    /// Canonical form of the watched path.
    ///
    /// Stored in addition to mtime to account for symlinked configs where
    /// the symlink target may change without an mtime change. Common on nix
    /// where everything links into /nix/store, which keeps no mtime
    /// (= 1970-01-01).
    canonical: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
enum CheckResult {
    Missing,
    Unchanged,
    Changed,
}

impl Watcher {
    pub fn new(
        path: ConfigPath,
        includes: Vec<PathBuf>,
        mut process: impl FnMut(&ConfigPath) -> ConfigParseResult<Config, ()> + Send + 'static,
        changed: SyncSender<Result<Config, ()>>,
    ) -> Self {
        let (load_config, load_config_rx) = mpsc::channel();

        thread::Builder::new()
            .name(format!("Filesystem Watcher for {path:?}"))
            .spawn(move || {
                let mut inner = WatcherInner::new(path, includes);

                loop {
                    // Doubles as the poll timer and the exit signal: the recv
                    // disconnects when the `Watcher` handle drops.
                    let mut should_load = match load_config_rx.recv_timeout(POLLING_INTERVAL) {
                        Ok(path) => {
                            if let Some(path) = path {
                                inner = WatcherInner::new(
                                    ConfigPath::Explicit(PathBuf::from(path)),
                                    Vec::new(),
                                );
                            }
                            true
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => false,
                    };

                    match inner.check() {
                        CheckResult::Missing => continue,
                        CheckResult::Unchanged => (),
                        CheckResult::Changed => {
                            tracing::debug!("config file changed");
                            should_load = true;
                        }
                    }

                    if should_load {
                        let res = process(&inner.path);

                        if let Err(err) = changed.send(res.config) {
                            tracing::warn!("error sending change notification: {err:?}");
                            break;
                        }

                        // There's a window between reading the config and
                        // stat'ing the includes where an included file could
                        // change unnoticed. No good way around it: the final
                        // include set is only known after the parse.
                        inner.set_includes(res.includes);
                    }
                }

                tracing::debug!("exiting watcher thread for {:?}", inner.path);
            })
            .unwrap();

        Self { load_config }
    }

    /// A detached force-reload trigger, for wiring into
    /// [`PrismState::config_load_request`] (the `load-config-file`
    /// action). Owns a clone of the trigger channel; calling it after
    /// the watcher thread exits is a silent no-op.
    ///
    /// [`PrismState::config_load_request`]: prism_protocols::PrismState
    pub fn loader(&self) -> Box<dyn Fn(Option<String>)> {
        let tx = self.load_config.clone();
        Box::new(move |path| {
            let _ = tx.send(path);
        })
    }
}

impl Props {
    fn from_path(path: &Path) -> io::Result<Self> {
        let canonical = path.canonicalize()?;
        let mtime = canonical.metadata()?.modified()?;
        Ok(Self { mtime, canonical })
    }

    fn from_config_path(config_path: &ConfigPath) -> io::Result<Self> {
        match config_path {
            ConfigPath::Explicit(path) => Self::from_path(path),
            ConfigPath::Regular {
                user_path,
                system_path,
            } => Self::from_path(user_path).or_else(|_| Self::from_path(system_path)),
        }
    }
}

impl WatcherInner {
    fn new(path: ConfigPath, includes: Vec<PathBuf>) -> Self {
        let last_props = Props::from_config_path(&path).ok();

        let mut rv = Self {
            path,
            last_props,
            includes: HashMap::new(),
        };
        rv.set_includes(includes);
        rv
    }

    fn check(&mut self) -> CheckResult {
        if let Ok(new_props) = Props::from_config_path(&self.path) {
            if self.last_props.as_ref() != Some(&new_props) {
                self.last_props = Some(new_props);
                CheckResult::Changed
            } else {
                for (path, last_props) in &mut self.includes {
                    let new_props = Props::from_path(path).ok();

                    // If an include goes missing while the main config file
                    // is unchanged, that still counts as a change: the parse
                    // result (likely an error) differs.
                    if *last_props != new_props {
                        return CheckResult::Changed;
                    }
                }

                CheckResult::Unchanged
            }
        } else {
            CheckResult::Missing
        }
    }

    fn set_includes(&mut self, includes: Vec<PathBuf>) {
        self.includes = includes
            .into_iter()
            .map(|path| {
                let props = Props::from_path(&path).ok();
                (path, props)
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;

    use super::*;

    /// One watcher over an explicit path inside a fresh temp dir.
    struct TestSetup {
        dir: tempfile::TempDir,
        watcher: WatcherInner,
    }

    impl TestSetup {
        /// Sets up `<tmp>/prism/config.kdl` as the watched path. `setup`
        /// runs first (gets the temp root) to create initial state.
        fn new(setup: impl FnOnce(&Path)) -> Self {
            let dir = tempfile::tempdir().unwrap();
            setup(dir.path());

            let config_path = ConfigPath::Explicit(dir.path().join("prism/config.kdl"));
            let includes = match &config_path {
                ConfigPath::Explicit(p) if p.exists() => Config::load(p).includes,
                _ => Vec::new(),
            };
            let mut rv = Self {
                dir,
                watcher: WatcherInner::new(config_path, includes),
            };
            // Nothing should trigger before the test acts; also ensures the
            // next write lands on a different mtime tick.
            rv.assert_unchanged();
            rv
        }

        fn root(&self) -> &Path {
            self.dir.path()
        }

        fn write(&self, rel: &str, content: &str) {
            let path = self.root().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }

        /// Ensures mtime differs between writes (mtime granularity).
        fn pass_time(&self) {
            thread::sleep(Duration::from_millis(50));
        }

        fn assert_unchanged(&mut self) {
            let res = self.watcher.check();
            // May be Missing or Unchanged, both are fine.
            assert_ne!(
                res,
                CheckResult::Changed,
                "watcher should not have noticed any changes"
            );
            self.pass_time();
        }

        fn assert_changed_to(&mut self, expected: &str) {
            let res = self.watcher.check();
            assert_eq!(
                res,
                CheckResult::Changed,
                "watcher should have noticed a change, but it didn't"
            );

            let ConfigPath::Explicit(path) = &self.watcher.path else {
                unreachable!()
            };
            let actual = fs::read_to_string(path).unwrap();
            assert_eq!(actual, expected, "wrong file contents");

            let includes = Config::load(path).includes;
            self.watcher.set_includes(includes);
            self.pass_time();
        }
    }

    #[test]
    fn change_file() {
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            fs::write(root.join("prism/config.kdl"), "// a").unwrap();
        });
        t.write("prism/config.kdl", "// b");
        t.assert_changed_to("// b");
        t.assert_unchanged();
    }

    #[test]
    fn overwrite_but_dont_change_file() {
        // A rewrite with identical contents still counts (mtime changed);
        // reloading an identical config is harmless.
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            fs::write(root.join("prism/config.kdl"), "// a").unwrap();
        });
        t.write("prism/config.kdl", "// a");
        t.assert_changed_to("// a");
    }

    #[test]
    fn create_file_later() {
        // Watched path doesn't exist at startup; creating it triggers.
        let mut t = TestSetup::new(|_| {});
        t.write("prism/config.kdl", "// a");
        t.assert_changed_to("// a");
        t.assert_unchanged();
    }

    #[test]
    fn remove_then_recreate_file() {
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            fs::write(root.join("prism/config.kdl"), "// a").unwrap();
        });
        fs::remove_file(t.root().join("prism/config.kdl")).unwrap();
        // Missing is not a change (keep the loaded config).
        t.assert_unchanged();
        t.write("prism/config.kdl", "// b");
        t.assert_changed_to("// b");
    }

    #[test]
    fn swap_symlink_target_without_mtime() {
        // The nix case: both targets keep mtime at the epoch; only the
        // canonical path changes.
        let epoch = |path: &Path, content: &str| {
            fs::write(path, content).unwrap();
            let f = fs::File::open(path).unwrap();
            f.set_times(
                fs::FileTimes::new()
                    .set_accessed(SystemTime::UNIX_EPOCH)
                    .set_modified(SystemTime::UNIX_EPOCH),
            )
            .unwrap();
        };
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            epoch(&root.join("prism/config2.kdl"), "// a");
            epoch(&root.join("prism/config3.kdl"), "// b");
            symlink("config2.kdl", root.join("prism/config.kdl")).unwrap();
        });
        let link = t.root().join("prism/config.kdl");
        fs::remove_file(&link).unwrap();
        symlink("config3.kdl", &link).unwrap();
        t.assert_changed_to("// b");
        t.assert_unchanged();
    }

    #[test]
    fn change_included_file() {
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            fs::write(root.join("prism/config.kdl"), "include \"colors.kdl\"").unwrap();
            fs::write(root.join("prism/colors.kdl"), "// colors").unwrap();
        });
        t.write("prism/colors.kdl", "// updated colors");
        t.assert_changed_to("include \"colors.kdl\"");
        t.assert_unchanged();
    }

    #[test]
    fn remove_included_file() {
        let mut t = TestSetup::new(|root| {
            fs::create_dir_all(root.join("prism")).unwrap();
            fs::write(root.join("prism/config.kdl"), "include \"colors.kdl\"").unwrap();
            fs::write(root.join("prism/colors.kdl"), "// colors").unwrap();
        });
        fs::remove_file(t.root().join("prism/colors.kdl")).unwrap();
        t.assert_changed_to("include \"colors.kdl\"");
    }
}
