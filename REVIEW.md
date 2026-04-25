# EEG Command Executor Refactor Review

## Overview
This document compares the original monolithic implementation (`eeg-preprc-v4-og.py`) with the new modular codebase (`main.py` and modules in `src/`). It focuses on the classification of the "Focus" mental state and highlights key differences, regressions, and recommendations.

---

## 1. Focus Classification Logic: Monolith vs. Modular

### Monolith (`eeg-preprc-v4-og.py`)
- **State Smoothing:** Uses a `state_history` deque for smoothing, matching the "ball game" logic (majority voting over recent states).
- **Thresholds:** Focus is detected if `beta_avg > alpha_avg * 1.3` and `beta_avg > 0.5` (or adaptive threshold if calibrated).
- **Calibration:** Supports user calibration to adapt thresholds for focus and relax states.
- **Immediate Key Handling:** When focus is detected, the space key is held; released when focus is lost.
- **Fallback:** If not focusing, always releases the space key.

### Modular Code (Current)
- **State Smoothing:** Also uses a `state_history` deque and majority voting for stability (restored from the monolith).
- **Thresholds:** Uses config-driven ratios (e.g., `focus_beta_alpha_ratio`) and a minimum beta value, but may differ from the monolith if config is not set identically.
- **Calibration:** Calibration manager is present, but the actual data collection and threshold update for focus/relax is not fully implemented (stubbed in key handler).
- **Key Handling:** Similar logic for holding/releasing the space key, but may be affected by differences in classification or missing calibration.
- **Fallback:** Same fallback logic, but may be less robust if calibration is not used.

---

## 2. Key Differences & Potential Issues

- **Calibration Workflow:**
  - **Monolith:** Fully implements calibration for focus/relax, updating thresholds based on user data.
  - **Modular:** CalibrationManager exists, but focus/relax calibration is not functional—thresholds remain static unless manually changed in config.

- **Configurable Parameters:**
  - **Monolith:** Thresholds and ratios are hardcoded but can be updated via calibration.
  - **Modular:** Thresholds and ratios are loaded from config, but if calibration is not run, they may not match the monolith's tuned values.

- **State Smoothing:**
  - Both use majority voting, but the modular version's effectiveness depends on the correct config and calibration.

- **Key Command Timing:**
  - Both versions send/release key commands based on state transitions, but the modular version may be more sensitive to jitter if classification is less robust.

- **Debugging/Logging:**
  - The modular version has more error handling, but both rely on print statements rather than structured logging.

---

## 3. Recommendations

1. **Implement Full Calibration:**
   - Complete the calibration workflow for focus/relax in the modular code so thresholds are adapted to each user/session.
2. **Verify Config Values:**
   - Ensure `focus_beta_alpha_ratio` and minimum beta thresholds in config match the monolith's logic for best results.
3. **Test with Real Data:**
   - Compare classification output (focus/neutral/relax) for the same EEG data in both versions to identify discrepancies.
4. **Add Logging:**
   - Use structured logging to debug why focus is not being detected (log alpha, beta, thresholds, and classification decisions).
5. **Unit Tests:**
   - Add tests for the classification function with synthetic data to ensure correct behavior.

---

## 4. Summary Table

| Feature                | Monolith           | Modular (Current)      |
|-----------------------|--------------------|------------------------|
| State smoothing       | Yes                | Yes                    |
| Focus threshold logic | Hardcoded/calibrated| Config-driven/static   |
| Calibration           | Fully implemented  | Stub only              |
| Logging               | Print              | Print                  |
| Key handling          | Yes                | Yes                    |

---

## 5. Conclusion

The modular refactor preserves most of the monolith's structure, but the lack of a working calibration workflow for focus/relax thresholds is likely causing issues with focus classification. Restoring and testing calibration, and ensuring config parity, should resolve most discrepancies.
