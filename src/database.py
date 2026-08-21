"""Database layer for user management, submissions, and version tracking."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, Float, Integer, String, Text, DateTime, Boolean, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()

PROJECT_VERSION = "1.0.0"


def real_leaderboard_score(valid_mcc: float | None, test_mcc: float | None) -> float:
    """Official Real leaderboard score: the lower of validation and test MCC."""
    valid = float(valid_mcc) if valid_mcc is not None else 0.0
    test = float(test_mcc) if test_mcc is not None else 0.0
    if not math.isfinite(valid):
        valid = 0.0
    if not math.isfinite(test):
        test = 0.0
    return min(valid, test)


class User(Base):
    """User account."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class Submission(Base):
    """Code submission for real leaderboard."""

    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False, index=True)
    dataset = Column(String(255), nullable=False, index=True)
    submission_name = Column(String(255), nullable=False)
    correction_code = Column(Text, nullable=False)
    model_code = Column(Text, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False, index=True)
    version_created = Column(String(32), default=PROJECT_VERSION, nullable=False)

    def __repr__(self) -> str:
        return f"<Submission {self.id} by {self.username} on {self.dataset}>"


class Score(Base):
    """Evaluation score for a submission."""

    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, nullable=False, index=True)
    test_mcc = Column(Float, nullable=True, default=0.0)  # Matthews Correlation Coefficient (primary metric)
    valid_mcc = Column(Float, nullable=True, default=0.0)
    valid_mcc_folds_json = Column(Text, nullable=True, default="[]")
    train_mcc = Column(Float, nullable=True, default=0.0)
    accuracy = Column(Float, nullable=False)
    macro_f1 = Column(Float, nullable=False)
    n_samples = Column(Integer, nullable=False)
    log_loss = Column(Float, nullable=True, default=None)
    brier_score = Column(Float, nullable=True, default=None)
    ece = Column(Float, nullable=True, default=None)
    batch_silhouette = Column(Float, nullable=True, default=None)
    batch_centroid_dispersion = Column(Float, nullable=True, default=None)
    batch_nbe = Column(Float, nullable=True, default=None)
    batch_nmi = Column(Float, nullable=True, default=None)
    batch_nri = Column(Float, nullable=True, default=None)
    version_evaluated = Column(String(32), default=PROJECT_VERSION, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    needs_recalc = Column(Boolean, default=False, nullable=False)
    plots_json = Column(Text, nullable=True, default="")  # Store visualization plots

    def __repr__(self) -> str:
        return f"<Score sub_id={self.submission_id} test_mcc={self.test_mcc:.4f} v{self.version_evaluated}>"


class VersionHistory(Base):
    """Track version changes and evaluation metadata."""

    __tablename__ = "version_history"

    id = Column(Integer, primary_key=True)
    version = Column(String(32), nullable=False, unique=True, index=True)
    released_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    major_version_bump = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<VersionHistory v{self.version}>"


class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, db_path: str | Path = "data/leaderboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate_legacy_schema()
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _migrate_legacy_schema(self) -> None:
        """Backfill columns for older SQLite databases created before schema updates."""
        with self.engine.begin() as conn:
            table_rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            table_names = {row[0] for row in table_rows}
            if "scores" not in table_names:
                return

            # SQLite table_info returns rows where index 1 is the column name.
            existing_columns = {
                row[1] for row in conn.execute(text("PRAGMA table_info(scores)"))
            }

            missing_columns = {
                "test_mcc": "ALTER TABLE scores ADD COLUMN test_mcc FLOAT DEFAULT 0.0",
                "valid_mcc": "ALTER TABLE scores ADD COLUMN valid_mcc FLOAT DEFAULT 0.0",
                "valid_mcc_folds_json": "ALTER TABLE scores ADD COLUMN valid_mcc_folds_json TEXT DEFAULT '[]'",
                "train_mcc": "ALTER TABLE scores ADD COLUMN train_mcc FLOAT DEFAULT 0.0",
                "needs_recalc": "ALTER TABLE scores ADD COLUMN needs_recalc BOOLEAN NOT NULL DEFAULT 0",
                "plots_json": "ALTER TABLE scores ADD COLUMN plots_json TEXT DEFAULT ''",
                "log_loss": "ALTER TABLE scores ADD COLUMN log_loss FLOAT DEFAULT NULL",
                "brier_score": "ALTER TABLE scores ADD COLUMN brier_score FLOAT DEFAULT NULL",
                "ece": "ALTER TABLE scores ADD COLUMN ece FLOAT DEFAULT NULL",
                "batch_silhouette": "ALTER TABLE scores ADD COLUMN batch_silhouette FLOAT DEFAULT NULL",
                "batch_centroid_dispersion": "ALTER TABLE scores ADD COLUMN batch_centroid_dispersion FLOAT DEFAULT NULL",
                "batch_nbe": "ALTER TABLE scores ADD COLUMN batch_nbe FLOAT DEFAULT NULL",
                "batch_nmi": "ALTER TABLE scores ADD COLUMN batch_nmi FLOAT DEFAULT NULL",
                "batch_nri": "ALTER TABLE scores ADD COLUMN batch_nri FLOAT DEFAULT NULL",
            }

            for col_name, alter_sql in missing_columns.items():
                if col_name not in existing_columns:
                    conn.execute(text(alter_sql))

    def get_session(self) -> Session:
        return self.SessionLocal()

    def get_or_create_user(self, username: str) -> User:
        session = self.get_session()
        user = session.query(User).filter_by(username=username).first()
        if not user:
            user = User(username=username)
            session.add(user)
            session.commit()
        session.close()
        return user

    def create_submission(
        self,
        username: str,
        dataset: str,
        submission_name: str,
        correction_code: str,
        model_code: str,
        is_public: bool = False,
        created_at: datetime | None = None,
        version_created: str | None = None,
    ) -> Submission:
        session = self.get_session()
        self.get_or_create_user(username)
        submission = Submission(
            username=username,
            dataset=dataset,
            submission_name=submission_name,
            correction_code=correction_code,
            model_code=model_code,
            is_public=is_public,
            created_at=created_at or datetime.now(timezone.utc),
            version_created=version_created or PROJECT_VERSION,
        )
        session.add(submission)
        session.commit()
        sub_id = submission.id
        session.close()
        return submission

    def create_score(
        self,
        submission_id: int,
        accuracy: float,
        macro_f1: float,
        n_samples: int,
        test_mcc: float = 0.0,
        valid_mcc: float = 0.0,
        valid_mcc_folds: list[float] | None = None,
        train_mcc: float = 0.0,
        log_loss: float | None = None,
        brier_score: float | None = None,
        ece: float | None = None,
        batch_silhouette: float | None = None,
        batch_centroid_dispersion: float | None = None,
        batch_nbe: float | None = None,
        batch_nmi: float | None = None,
        batch_nri: float | None = None,
        version: str | None = None,
        plots_json: str = "",
        created_at: datetime | None = None,
    ) -> Score:
        session = self.get_session()
        score = Score(
            submission_id=submission_id,
            test_mcc=float(test_mcc),
            valid_mcc=float(valid_mcc),
            valid_mcc_folds_json=json.dumps(
                [float(value) for value in (valid_mcc_folds or [])]
            ),
            train_mcc=float(train_mcc),
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            n_samples=int(n_samples),
            log_loss=float(log_loss) if log_loss is not None else None,
            brier_score=float(brier_score) if brier_score is not None else None,
            ece=float(ece) if ece is not None else None,
            batch_silhouette=float(batch_silhouette) if batch_silhouette is not None else None,
            batch_centroid_dispersion=float(batch_centroid_dispersion) if batch_centroid_dispersion is not None else None,
            batch_nbe=float(batch_nbe) if batch_nbe is not None else None,
            batch_nmi=float(batch_nmi) if batch_nmi is not None else None,
            batch_nri=float(batch_nri) if batch_nri is not None else None,
            version_evaluated=version or PROJECT_VERSION,
            created_at=created_at or datetime.now(timezone.utc),
            needs_recalc=False,
            plots_json=plots_json,
        )
        session.add(score)
        session.commit()
        session.close()
        return score

    def cleanup_old_submissions(self, dataset: str, limit: int = 100) -> int:
        """Keep only top N submissions per dataset, ranked by official Real score."""
        session = self.get_session()
        rows = (
            session.query(Submission.id)
            .join(Score, Submission.id == Score.submission_id)
            .filter(Submission.dataset == dataset)
            .distinct()
            .all()
        )
        scored_rows = []
        all_ids = [sub_id for sub_id, in rows]
        for sub_id, in rows:
            latest = (
                session.query(Score)
                .filter_by(submission_id=sub_id)
                .order_by(Score.created_at.desc())
                .first()
            )
            if latest is not None:
                scored_rows.append(
                    (
                        real_leaderboard_score(latest.valid_mcc, latest.test_mcc),
                        float(latest.accuracy or 0.0),
                        sub_id,
                    )
                )
        scored_rows.sort(reverse=True)
        top_ids = [sub_id for _, _, sub_id in scored_rows[:limit]]
        delete_ids = [sub_id for sub_id in all_ids if sub_id not in set(top_ids)]
        
        if not top_ids:
            session.close()
            return 0

        # Delete submissions NOT in the top IDs
        deleted_count = (
            session.query(Submission)
            .filter(Submission.dataset == dataset)
            .filter(Submission.id.in_(delete_ids))
            .delete(synchronize_session=False)
        )
        
        # Delete scores too
        if delete_ids:
            session.query(Score).filter(Score.submission_id.in_(delete_ids)).delete(synchronize_session=False)
        
        session.commit()
        session.close()
        return deleted_count

    def get_latest_score(self, submission_id: int) -> Optional[Score]:
        session = self.get_session()
        score = (
            session.query(Score)
            .filter_by(submission_id=submission_id)
            .order_by(Score.created_at.desc())
            .first()
        )
        session.close()
        return score

    def get_submissions_for_user(self, username: str) -> list[Submission]:
        session = self.get_session()
        submissions = session.query(Submission).filter_by(username=username).all()
        session.close()
        return submissions

    def get_public_submissions(self, dataset: str | None = None) -> list[Submission]:
        session = self.get_session()
        query = session.query(Submission).filter_by(is_public=True)
        if dataset:
            query = query.filter_by(dataset=dataset)
        submissions = query.all()
        session.close()
        return submissions

    def get_submission_by_id(self, submission_id: int) -> Optional[Submission]:
        session = self.get_session()
        submission = session.query(Submission).filter_by(id=submission_id).first()
        session.close()
        return submission

    def get_leaderboard(self, dataset: str | None = None) -> list[dict]:
        """Get leaderboard sorted by the lower of validation MCC and test MCC."""
        session = self.get_session()
        query = session.query(Submission, Score).join(
            Score, Submission.id == Score.submission_id
        )
        if dataset:
            query = query.filter(Submission.dataset == dataset)

        results = query.all()

        leaderboard = []
        for sub, score in results:
            official_score = real_leaderboard_score(score.valid_mcc, score.test_mcc)
            leaderboard.append(
                {
                    "submission_id": sub.id,
                    "username": sub.username,
                    "dataset": sub.dataset,
                    "submission_name": sub.submission_name,
                    "score": official_score,
                    "test_mcc": score.test_mcc,
                    "valid_mcc": score.valid_mcc,
                    "valid_mcc_folds": " | ".join(
                        f"{float(value):.4f}"
                        for value in json.loads(score.valid_mcc_folds_json or "[]")
                    ),
                    "train_mcc": score.train_mcc,
                    "accuracy": score.accuracy,
                    "macro_f1": score.macro_f1,
                    "n_samples": score.n_samples,
                    "log_loss": score.log_loss,
                    "brier_score": score.brier_score,
                    "ece": score.ece,
                    "batch_silhouette": score.batch_silhouette,
                    "batch_centroid_dispersion": score.batch_centroid_dispersion,
                    "batch_nbe": score.batch_nbe,
                    "batch_nmi": score.batch_nmi,
                    "batch_nri": score.batch_nri,
                    "created_at": sub.created_at.isoformat(),
                    "version_created": sub.version_created,
                    "version_evaluated": score.version_evaluated,
                    "is_public": sub.is_public,
                    "correction_code": sub.correction_code,
                    "model_code": sub.model_code,
                    "plots_json": score.plots_json,  # Visualization data (private storage)
                }
            )
        session.close()
        leaderboard.sort(
            key=lambda row: (
                float(row.get("score") or 0.0),
                float(row.get("accuracy") or 0.0),
            ),
            reverse=True,
        )
        return leaderboard

    def mark_for_recalculation(self, old_version: str) -> int:
        """Mark all scores from a previous version for recalculation."""
        session = self.get_session()
        count = (
            session.query(Score)
            .filter(Score.version_evaluated == old_version)
            .update({"needs_recalc": True})
        )
        session.commit()
        session.close()
        return count

    def get_submissions_needing_recalc(self) -> list[int]:
        """Get all submission IDs that need score recalculation."""
        session = self.get_session()
        submissions = session.query(Score.submission_id).filter(Score.needs_recalc == True).distinct().all()
        session.close()
        return [s[0] for s in submissions]

    def record_version(self, version: str, is_major: bool = False, notes: str = "") -> VersionHistory:
        """Record a version release."""
        session = self.get_session()
        vh = VersionHistory(
            version=version,
            major_version_bump=is_major,
            notes=notes,
        )
        session.add(vh)
        session.commit()
        session.close()
        return vh

    def get_current_version(self) -> str:
        """Get the current project version."""
        return PROJECT_VERSION
