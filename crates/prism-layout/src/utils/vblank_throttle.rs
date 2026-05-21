//! VBlank throttling.
//!
//! Some buggy drivers deliver VBlanks way earlier than necessary. This helper throttles the VBlank
//! in such cases to avoid tearing and to get more consistent timings.
//!
//! Generic over the calloop loop's state type — niri held one `State`, prism
//! threads `PrismState` through the same way. Functionally identical to
//! niri's version.

use std::time::Duration;

use calloop::timer::{TimeoutAction, Timer};
use calloop::{LoopHandle, RegistrationToken};
use tracing::warn;

#[derive(Debug)]
pub struct VBlankThrottle<S: 'static> {
    event_loop: LoopHandle<'static, S>,
    last_vblank_timestamp: Option<Duration>,
    throttle_timer_token: Option<RegistrationToken>,
    printed_warning: bool,
    output_name: String,
}

impl<S: 'static> VBlankThrottle<S> {
    pub fn new(event_loop: LoopHandle<'static, S>, output_name: String) -> Self {
        Self {
            event_loop,
            last_vblank_timestamp: None,
            throttle_timer_token: None,
            printed_warning: false,
            output_name,
        }
    }

    pub fn throttle(
        &mut self,
        refresh_interval: Option<Duration>,
        timestamp: Duration,
        mut call_vblank: impl FnMut(&mut S) + 'static,
    ) -> bool {
        if let Some(token) = self.throttle_timer_token.take() {
            self.event_loop.remove(token);
        }

        if let Some((last, refresh)) = Option::zip(self.last_vblank_timestamp, refresh_interval) {
            let passed = timestamp.saturating_sub(last);
            if passed < refresh / 2 {
                if !self.printed_warning {
                    self.printed_warning = true;
                    warn!(
                        "output {} running faster than expected, throttling vblanks: \
                         expected refresh {refresh:?}, got vblank after {passed:?}",
                        self.output_name
                    );
                }

                let remaining = refresh - passed;
                let token = self
                    .event_loop
                    .insert_source(Timer::from_duration(remaining), move |_, _, state| {
                        call_vblank(state);
                        TimeoutAction::Drop
                    })
                    .unwrap();
                self.throttle_timer_token = Some(token);
                return true;
            }
        }

        self.last_vblank_timestamp = Some(timestamp);
        false
    }
}
