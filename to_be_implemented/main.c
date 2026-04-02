// CS528 Air Drawing Project

/**
 * USB Serial initially -> move to tcp server for wireless streaming for future milestones
 * Wiring:
 *   MPU-6050 VCC  -> 3.3V
 *   MPU-6050 GND  -> GND
 *   MPU-6050 SCL  -> GPIO 19 (same as hw1 and hw2 for starting)
 *   MPU-6050 SDA  -> GPIO 18 (same as hw1 and hw2 for starting)
 *   MPU-6050 AD0  -> GND  (sets I2C addr = 0x68)
 *
 * Output format (CSV, 100Hz):
 *   ts_ms, ax, ay, az, gx, gy, gz,
 *   pitch, roll,      <- orientation angles (degrees) from complementary filter
 *   lax, lay, laz,    <- linear accel (gravity removed)
 *   mag,              <- accel magnitude for stroke detection
 *   stroke            <- 0=idle 1=active 2=start 3=end
 */

#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "signal_proc.h"

/* ── I2C config ─────────────────────────────────────────── */
#define I2C_MASTER_NUM      I2C_NUM_0
#define I2C_MASTER_SDA_IO   18              // GPIO 18 (same as HW1/HW2)
#define I2C_MASTER_SCL_IO   19              // GPIO 19 (same as HW1/HW2)
#define I2C_MASTER_FREQ_HZ  400000          // 400 kHz fast mode
#define I2C_TIMEOUT_MS      100

/* ── MPU-6050 register map ───────────────────────────────── */
#define MPU6050_ADDR        0x68            // AD0 = GND
#define MPU_REG_PWR_MGMT_1  0x6B
#define MPU_REG_SMPLRT_DIV  0x19
#define MPU_REG_CONFIG      0x1A
#define MPU_REG_GYRO_CFG    0x1B
#define MPU_REG_ACCEL_CFG   0x1C
#define MPU_REG_ACCEL_XOUT  0x3B           // 6 bytes: AX_H AX_L AY_H AY_L AZ_H AZ_L
#define MPU_REG_GYRO_XOUT   0x43           // 6 bytes: GX_H GX_L GY_H GY_L GZ_H GZ_L
#define MPU_REG_WHO_AM_I    0x75

/* ── Scale factors ───────────────────────────────────────── */
// Accel +/-2g  -> LSB/g = 16384
// Gyro  +/-250 degrees/s -> LSB/(degrees/s) = 131
#define ACCEL_SCALE  16384.0f
#define GYRO_SCALE   131.0f
#define SAMPLE_DT    0.01f                  // 10ms = 100 Hz

static const char *TAG = "MPU6050";

/* ─────────────────────────────────────────────────────────
 * Low-level I2C helpers
 * ───────────────────────────────────────────────────────── */
static esp_err_t i2c_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_master_write_to_device(
        I2C_MASTER_NUM, MPU6050_ADDR,
        buf, sizeof(buf),
        pdMS_TO_TICKS(I2C_TIMEOUT_MS));
}

static esp_err_t i2c_read_regs(uint8_t reg, uint8_t *out, size_t len)
{
    return i2c_master_write_read_device(
        I2C_MASTER_NUM, MPU6050_ADDR,
        &reg, 1,
        out, len,
        pdMS_TO_TICKS(I2C_TIMEOUT_MS));
}

/* ─────────────────────────────────────────────────────────
 * I2C bus init
 * ───────────────────────────────────────────────────────── */
static void i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_MASTER_SDA_IO,
        .scl_io_num       = I2C_MASTER_SCL_IO,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
    ESP_LOGI(TAG, "I2C master init OK (SDA=%d SCL=%d)", I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);
}

/* ─────────────────────────────────────────────────────────
 * MPU-6050 init sequence
 * ───────────────────────────────────────────────────────── */
static void mpu6050_init(void) // config sample rate, scale, etc -> check statements for debugging
{
    // Check WHO_AM_I — should return 0x68
    uint8_t who = 0;
    ESP_ERROR_CHECK(i2c_read_regs(MPU_REG_WHO_AM_I, &who, 1));
    if (who != 0x68) {
        ESP_LOGE(TAG, "WHO_AM_I mismatch: got 0x%02X, expected 0x68", who);
        // Don't hard-fail; continue anyway in case of clone chip
    } else {
        ESP_LOGI(TAG, "MPU-6050 detected (WHO_AM_I=0x%02X)", who);
    }

    // Wake up: clear SLEEP bit, use internal 8 MHz oscillator
    ESP_ERROR_CHECK(i2c_write_reg(MPU_REG_PWR_MGMT_1, 0x00));
    vTaskDelay(pdMS_TO_TICKS(10));  // datasheet: 10ms after wake-up

    // Sample rate divider: SMPLRT_DIV = 9 -> 1000/(1+9) = 100 Hz
    ESP_ERROR_CHECK(i2c_write_reg(MPU_REG_SMPLRT_DIV, 9));

    // DLPF: bandwidth ~44 Hz accel / 42 Hz gyro (CONFIG = 3)
    ESP_ERROR_CHECK(i2c_write_reg(MPU_REG_CONFIG, 0x03));

    // Gyro full scale: +/-250 degrees/s (bits [4:3] = 00)
    ESP_ERROR_CHECK(i2c_write_reg(MPU_REG_GYRO_CFG, 0x00));

    // Accel full scale: +/-2g (bits [4:3] = 00)
    ESP_ERROR_CHECK(i2c_write_reg(MPU_REG_ACCEL_CFG, 0x00));

    ESP_LOGI(TAG, "MPU-6050 configured: 100 Hz, +/-2g, +/-250 dps, DLPF=44Hz");
}

/* ─────────────────────────────────────────────────────────
 * Read one sample: accel (g) + gyro (degrees/s)
 * ───────────────────────────────────────────────────────── */
typedef struct {
    float ax, ay, az;   // g
    float gx, gy, gz;   // degrees/s
} imu_sample_t;

static esp_err_t mpu6050_read(imu_sample_t *s)
{
    uint8_t raw[14];
    // Burst read 14 bytes starting at ACCEL_XOUT_H
    // Layout: AX(2) AY(2) AZ(2) TEMP(2) GX(2) GY(2) GZ(2)
    esp_err_t err = i2c_read_regs(MPU_REG_ACCEL_XOUT, raw, 14);
    if (err != ESP_OK) return err;

    int16_t ax_raw = (int16_t)((raw[0]  << 8) | raw[1]);
    int16_t ay_raw = (int16_t)((raw[2]  << 8) | raw[3]);
    int16_t az_raw = (int16_t)((raw[4]  << 8) | raw[5]);
    // raw[6..7] = temperature, skip
    int16_t gx_raw = (int16_t)((raw[8]  << 8) | raw[9]);
    int16_t gy_raw = (int16_t)((raw[10] << 8) | raw[11]);
    int16_t gz_raw = (int16_t)((raw[12] << 8) | raw[13]);

    s->ax = ax_raw / ACCEL_SCALE;
    s->ay = ay_raw / ACCEL_SCALE;
    s->az = az_raw / ACCEL_SCALE;
    s->gx = gx_raw / GYRO_SCALE;
    s->gy = gy_raw / GYRO_SCALE;
    s->gz = gz_raw / GYRO_SCALE;

    return ESP_OK;
}

/* ─────────────────────────────────────────────────────────
 * Stroke state encoding
 * 0=idle  1=active  2=start pulse  3=end pulse
 * ───────────────────────────────────────────────────────── */
static inline int stroke_code(const sp_state_t *s)
{
    if (s->stroke_start)  return 2;
    if (s->stroke_end)    return 3;
    if (s->stroke_active) return 1;
    return 0;
}

/* ─────────────────────────────────────────────────────────
 * Main task: read + signal process + print CSV at ~100 Hz
 * ───────────────────────────────────────────────────────── */
static void imu_stream_task(void *arg)
{
    imu_sample_t raw;
    sp_state_t   sp;

    // Take one reading first to seed the complementary filter
    // so pitch/roll start at the actual resting orientation,
    // not zero. Retry until the sensor responds.
    esp_err_t seed_err;
    do {
        seed_err = mpu6050_read(&raw);
        vTaskDelay(pdMS_TO_TICKS(5));
    } while (seed_err != ESP_OK);

    sp_init(&sp, raw.ax, raw.ay, raw.az);

    const TickType_t period = pdMS_TO_TICKS(10);  // 10ms = 100 Hz
    TickType_t last_wake = xTaskGetTickCount();

    // Print CSV header so Python can parse by column name
    printf("ts_ms,ax,ay,az,gx,gy,gz,pitch,roll,lax,lay,laz,mag,stroke\n");
    fflush(stdout);

    while (1) {
        esp_err_t err = mpu6050_read(&raw);

        if (err == ESP_OK) {
            sp_update(&sp,
                      raw.ax, raw.ay, raw.az,
                      raw.gx, raw.gy, raw.gz,
                      SAMPLE_DT);

            int64_t ts_ms = esp_timer_get_time() / 1000;
            printf("%lld,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                   "%.3f,%.3f,"
                   "%.4f,%.4f,%.4f,"
                   "%.4f,%d\n",
                   ts_ms,
                   raw.ax, raw.ay, raw.az,
                   raw.gx, raw.gy, raw.gz,
                   sp.pitch, sp.roll,
                   sp.lax, sp.lay, sp.laz,
                   sp.accel_mag,
                   stroke_code(&sp));
        } else {
            ESP_LOGE(TAG, "Read error: %s", esp_err_to_name(err));
        }

        vTaskDelayUntil(&last_wake, period);
    }
}

/* ─────────────────────────────────────────────────────────
 * App entry point
 * ───────────────────────────────────────────────────────── */
void app_main(void)
{
    ESP_LOGI(TAG, "CS528 Air Drawing — Milestone 1/2");
    i2c_master_init();
    mpu6050_init();
    xTaskCreate(imu_stream_task, "imu_stream", 4096, NULL, 5, NULL);
}