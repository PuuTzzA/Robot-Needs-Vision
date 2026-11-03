# Stage 2 Audio Clips Directory

All audio clips are located in this directory (`Stage2/stage2_system/laptop/audio/`).

The audio announcement engine uses a **composable modular design**. Instead of recording dozens of full sentences, announcements are assembled dynamically from **action lead-in clips** and **reusable token clips** (letters and digits).

If any clip is missing or has not yet been placed here, the system automatically falls back to offline Text-To-Speech (TTS) so announcements never fail.

---

## 1. Intro Clip (1 Clip)
- **`wazzup_beijing.mp3`** (or `whats_up_beijing.mp3`): Spoken script: *"What's up Beijing!"*
  (Played at the start of every level before the robot moves).

---

## 2. Action Lead-In Clips (8 Clips)
- **`starting_level.mp3`**: *"Starting Level"* (followed by level number, e.g. `one.mp3`)
- **`finished_level.mp3`**: *"Finished Level"* (followed by level number, e.g. `three.mp3`)
- **`going_to_point.mp3`**: *"Going to Point"* (followed by point letter, e.g. `b.mp3`)
- **`retrying_to_go_to_point.mp3`**: *"Retrying to go to Point"* (followed by point letter, e.g. `b.mp3`)
- **`failed_to_go_to_point.mp3`**: *"Failed to go to Point"* (followed by failed point letter, e.g. `b.mp3`)
- **`proceeding_to_point.mp3`**: *"proceeding to Point"* (followed by detour point letter, e.g. `d.mp3`)
- **`taking_photo_of_tasks.mp3`**: *"Taking photo of tasks"* (played when photographing the laptop screen at Point 0)
- **`taking_photo_of_point.mp3`**: *"Taking photo of Point"* (followed by point letter, e.g. `b.mp3`)
- **`returning_to_start.mp3`**: *"Returning to start position"* (played when parallel return to Start is dispatched)

---

## 3. Reusable Identifier Tokens (13 Clips)

### Digits (4 Clips)
- **`zero.mp3`** (or `0.mp3`): *"Zero"* (reused for Point 0)
- **`one.mp3`** (or `1.mp3`): *"One"* (reused for Level 1)
- **`two.mp3`** (or `2.mp3`): *"Two"* (reused for Level 2)
- **`three.mp3`** (or `3.mp3`): *"Three"* (reused for Level 3)

### Keypoint Letters (9 Clips)
Reused across going to, retrying, failing, proceeding to, and taking photo of point:
- **`a.mp3`**
- **`b.mp3`**
- **`c.mp3`**
- **`d.mp3`**
- **`e.mp3`**
- **`f.mp3`**
- **`g.mp3`**
- **`h.mp3`**
- **`i.mp3`**

*(Filenames are case-insensitive and support aliases).*

---

## Composition Examples

| Event | Composed Sequence | Resulting Spoken Audio |
| :--- | :--- | :--- |
| **Level 1 Start** | `wazzup_beijing.mp3` + `starting_level.mp3` + `one.mp3` | *"What's up Beijing! Starting Level One."* |
| **Move to Point 0** | `going_to_point.mp3` + `zero.mp3` | *"Going to Point Zero."* |
| **Capture Task Screen** | `taking_photo_of_tasks.mp3` | *"Taking photo of tasks."* |
| **Visit Point B** | `going_to_point.mp3` + `b.mp3` | *"Going to Point B."* |
| **Capture Point B** | `taking_photo_of_point.mp3` + `b.mp3` | *"Taking photo of Point B."* |
| **Failure Detour** | `failed_to_go_to_point.mp3` + `b.mp3` + `proceeding_to_point.mp3` + `d.mp3` | *"Failed to go to Point B, proceeding to Point D."* |
| **Retry Point B** | `retrying_to_go_to_point.mp3` + `b.mp3` | *"Retrying to go to Point B."* |
| **Return to Start** | `returning_to_start.mp3` | *"Returning to start position."* |
| **Finish Level 3** | `finished_level.mp3` + `three.mp3` + **Dynamic TTS Readout** | *"Finished Level Three. [Dynamic LLM sentence result]"* |
