-- GlucoSight database schema (MariaDB 10.4+ / MySQL 8). Applied by
-- `python -m integration.db.init_db`; every statement is idempotent.
-- Statements are split on ';', so keep semicolons out of comments and strings.

-- Accounts. `participant_id` is the pseudonymous id the prediction API and the
-- research pipeline see (contract field `participant_id`), so the email and
-- name never have to leave this table.
CREATE TABLE IF NOT EXISTS users (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    participant_id CHAR(36)        NOT NULL,
    name          VARCHAR(120)    NOT NULL,
    email         VARCHAR(254)    NOT NULL,
    password_hash VARCHAR(255)    NOT NULL,
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP       NULL DEFAULT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_email (email),
    UNIQUE KEY uq_users_participant (participant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Server-side login sessions. Only the SHA-256 of the cookie token is stored,
-- so a leaked table cannot be replayed as a login. Logging out deletes the row.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash CHAR(64)        NOT NULL,
    user_id    BIGINT UNSIGNED NOT NULL,
    created_at TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME        NOT NULL,
    user_agent VARCHAR(255)    NULL DEFAULT NULL,
    PRIMARY KEY (token_hash),
    KEY idx_sessions_user (user_id),
    KEY idx_sessions_expires (expires_at),
    CONSTRAINT fk_sessions_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Per-account profile (name and email stay on `users`). One row per user,
-- created on first save. Blank form fields are stored as NULL.
CREATE TABLE IF NOT EXISTS profiles (
    user_id        BIGINT UNSIGNED NOT NULL,
    age            TINYINT UNSIGNED NULL DEFAULT NULL,
    gender         VARCHAR(20)     NOT NULL DEFAULT 'Pria',
    height_cm      DECIMAL(4,1)    NULL DEFAULT NULL,
    weight_kg      DECIMAL(4,1)    NULL DEFAULT NULL,
    diabetes_type  VARCHAR(30)     NOT NULL DEFAULT 'Tipe 2',
    default_source VARCHAR(10)     NOT NULL DEFAULT 'meter',
    updated_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id),
    CONSTRAINT fk_profiles_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Prediction history shown on the Riwayat screen. `id` is the meal_id the
-- prediction API returned, so re-saving the same result is a no-op.
-- created_at is UTC.
CREATE TABLE IF NOT EXISTS predictions (
    user_id        BIGINT UNSIGNED NOT NULL,
    id             VARCHAR(64)     NOT NULL,
    created_at     DATETIME(3)     NOT NULL,
    food           VARCHAR(255)    NOT NULL,
    portion        VARCHAR(50)     NOT NULL,
    t60            FLOAT           NOT NULL,
    t120           FLOAT           NOT NULL,
    glucose_range  VARCHAR(12)     NOT NULL,
    g0             FLOAT           NOT NULL,
    source_id      VARCHAR(10)     NOT NULL,
    basis          TEXT            NOT NULL,
    usage_message  TEXT            NOT NULL,
    model_version  VARCHAR(100)    NOT NULL,
    PRIMARY KEY (user_id, id),
    KEY idx_predictions_user_created (user_id, created_at),
    CONSTRAINT fk_predictions_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
