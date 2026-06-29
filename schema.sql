-- PostgreSQL schema
-- Run: psql -d wc2026 -f schema.sql

CREATE TABLE matches (
    match_id        SERIAL PRIMARY KEY,
    match_date      DATE NOT NULL,
    home_team       VARCHAR(100) NOT NULL,
    away_team       VARCHAR(100) NOT NULL,
    home_score      INT,
    away_score      INT,
    tournament      VARCHAR(200),
    neutral         BOOLEAN DEFAULT FALSE,
    is_competitive  BOOLEAN DEFAULT TRUE,
    source          VARCHAR(50),   -- 'kaggle', 'api_football', 'live'
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE elo_ratings (
    id              SERIAL PRIMARY KEY,
    team            VARCHAR(100) NOT NULL,
    elo_rating      FLOAT NOT NULL,
    rating_date     DATE NOT NULL,
    match_id        INT REFERENCES matches(match_id),
    UNIQUE(team, rating_date)
);

CREATE TABLE fifa_rankings (
    id              SERIAL PRIMARY KEY,
    rank_date       DATE NOT NULL,
    country         VARCHAR(100) NOT NULL,
    rank_position   INT,
    total_points    FLOAT,
    confederation   VARCHAR(20),
    UNIQUE(rank_date, country)
);

CREATE TABLE squad_stats (
    id              SERIAL PRIMARY KEY,
    team            VARCHAR(100) NOT NULL,
    season          INT NOT NULL,
    competition     VARCHAR(200),
    total_value_eur FLOAT,
    avg_age         FLOAT,
    xg_per_game     FLOAT,
    xga_per_game    FLOAT,
    ppda            FLOAT,
    possession_pct  FLOAT,
    sot_pct         FLOAT,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE wc2026_schedule (
    match_id        VARCHAR(20) PRIMARY KEY,  -- e.g. "WC2026_001"
    round           VARCHAR(50),
    group_name      VARCHAR(5),
    match_date      DATE,
    kickoff_utc     TIMESTAMP WITH TIME ZONE,
    team_a          VARCHAR(100),
    team_b          VARCHAR(100),
    venue           VARCHAR(200),
    city            VARCHAR(100),
    country         VARCHAR(100),
    result_a        INT,             -- filled in post-match
    result_b        INT,
    played          BOOLEAN DEFAULT FALSE
);

CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    match_id        VARCHAR(20) REFERENCES wc2026_schedule(match_id),
    model_version   VARCHAR(50),     -- e.g. "v1.0", "v1.3-retrained"
    prob_team_a_win FLOAT,
    prob_draw       FLOAT,
    prob_team_b_win FLOAT,
    predicted_score_a INT,
    predicted_score_b INT,
    brier_score     FLOAT,           -- filled in post-match
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tournament_probabilities (
    id              SERIAL PRIMARY KEY,
    team            VARCHAR(100),
    model_version   VARCHAR(50),
    win_tournament  FLOAT,
    reach_final     FLOAT,
    reach_semi      FLOAT,
    reach_quarters  FLOAT,
    advance_groups  FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX idx_elo_team_date ON elo_ratings(team, rating_date);
CREATE INDEX idx_predictions_match ON predictions(match_id);