#pragma once
/**
 * orientation-based signal processing for air drawing:
 *   - complementary filter: combines accel + gyro for stable pitch/roll
 *   - stroke detection via accel magnitude (hysteresis + debounce)
 *
 * using orientation instead of position and velocity for air drawing:
 *   drawing in air = rotating your finger/wrist to aim.
 *   pitch and roll map naturally to canvas Y and X.
 *   one integration (gyro -> angle) vs two (accel -> vel -> pos)
 *   means far less drift per stroke.
 */

#include <stdint.h>
#include <stdbool.h>

// tunings params -> will adjust later for best performance

// complementary filter weight for gyro vs accel (0.0–1.0).
// higher = trust gyro more (smoother but drifts over time).
// lower  = trust accel more (stable long-term but noisier).
// 0.96 is a standard starting point at 100 Hz.
#define SP_COMP_ALPHA        0.96f
#define SP_STROKE_ON_THRESH  0.15f // stroke detection: accel magnitude (g) above this = pen is moving
// stroke detection: accel magnitude (g) below this = pen has stopped
// hysteresis gap prevents flickering at the boundary
#define SP_STROKE_OFF_THRESH 0.08f
#define SP_STROKE_ON_DEBOUNCE   3 // debouncing: needs to be above threshold for this amt of samples
#define SP_STROKE_OFF_DEBOUNCE  8 // more debounce on stroke OFF than ON to prevent chatter at end of strokes

// data types

typedef struct {
    // raw imu values
    float ax, ay, az; // accelerometer (g)
    float gx, gy, gz; // gyroscope (degrees/s)

    // complementary filter output — orientation angles (degrees)
    // pitch: finger tilted forward/back  -> maps to canvas Y
    // roll:  finger tilted left/right    -> maps to canvas X
    float pitch; // degrees, range ~ -90 to +90
    float roll; // degrees, range ~ -90 to +90

    // Linear acceleration (gravity removed) in g
    // Used only for stroke detection magnitude
    float lax, lay, laz;
    float accel_mag;

    // stroke state
    bool stroke_active; // true while pen is drawing
    bool stroke_start; // true for exactly ONE sample at stroke begin
    bool stroke_end; // true for exactly ONE sample at stroke finish

    // sample counter
    uint16_t sample_id;
} sp_state_t;

/* API */

// call once at startup — seeds pitch/roll from accel so the first sample doesn't jump
void sp_init(sp_state_t *s, float ax_g, float ay_g, float az_g);

// call every sample (100 Hz) with fresh IMU values, updates all fields in s in-place
// dt_s = time since last sample in seconds (0.01 at 100 Hz)
void sp_update(sp_state_t *s,
               float ax_g, float ay_g, float az_g,
               float gx_dps, float gy_dps, float gz_dps,
               float dt_s);