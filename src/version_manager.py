"""Version management and submission recalculation."""

from datetime import datetime
from typing import Optional

from sqlalchemy import select

from src.code_challenge import run_code_submission, CodeValidationError
from src.database import DatabaseManager, VersionHistory, Score, Submission


class VersionManager:
    """Manages project versioning and bulk recalculation."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def bump_version(
        self,
        new_version: str,
        is_major: bool = False,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Bump version and optionally mark old submissions for recalculation.

        Args:
            new_version: Version string (e.g., "1.1.0")
            is_major: If True, mark all submissions from old version for recalculation
            notes: Optional release notes

        Returns:
            Dictionary with version record and recalculation status
        """
        version_record = self.db.record_version(
            version=new_version,
            is_major=is_major,
            notes=notes,
        )

        recalc_count = 0
        if is_major:
            with self.db.get_session() as session:
                old_version = (
                    session.query(VersionHistory)
                    .filter(VersionHistory.version != new_version)
                    .order_by(VersionHistory.released_at.desc())
                    .first()
                )
                if old_version:
                    recalc_count = self.db.mark_for_recalculation(old_version.version)

        return {
            "new_version": new_version,
            "is_major": is_major,
            "submitted_at": version_record.released_at,
            "submissions_marked_for_recalc": recalc_count,
            "notes": notes,
        }

    def recalculate_submission(
        self,
        submission_id: int,
        new_version: str,
    ) -> dict:
        """
        Recalculate a single submission against the current code.

        Args:
            submission_id: ID of submission to recalculate
            new_version: New version string for the score

        Returns:
            Dictionary with recalculation results
        """
        with self.db.get_session() as session:
            submission = (
                session.query(Submission)
                .filter(Submission.id == submission_id)
                .first()
            )
            if not submission:
                return {"success": False, "error": f"Submission {submission_id} not found"}

            try:
                _, metrics, _, _ = run_code_submission(
                    team=submission.username,
                    model_name=submission.submission_name,
                    dataset=submission.dataset,
                    correction_code=submission.correction_code,
                    model_code=submission.model_code,
                )
            except (CodeValidationError, Exception) as exc:
                return {
                    "success": False,
                    "error": f"Recalculation failed: {type(exc).__name__}: {exc}",
                }

            old_score = (
                session.query(Score)
                .filter(Score.submission_id == submission_id)
                .order_by(Score.created_at.desc())
                .first()
            )

            new_score = self.db.create_score(
                submission_id=submission_id,
                accuracy=float(metrics["accuracy"]),
                macro_f1=float(metrics["macro_f1"]),
                n_samples=int(metrics["n_samples"]),
                version=new_version,
            )

            if old_score:
                old_score.needs_recalc = False
                session.commit()

            return {
                "success": True,
                "submission_id": submission_id,
                "username": submission.username,
                "dataset": submission.dataset,
                "submission_name": submission.submission_name,
                "old_accuracy": float(old_score.accuracy) if old_score else None,
                "new_accuracy": float(metrics["accuracy"]),
                "old_macro_f1": float(old_score.macro_f1) if old_score else None,
                "new_macro_f1": float(metrics["macro_f1"]),
                "recalculated_at": new_score.created_at,
            }

    def bulk_recalculate(self, new_version: str, max_count: Optional[int] = None) -> list[dict]:
        """
        Recalculate all submissions marked for recalculation.

        Args:
            new_version: Version string to apply to new scores
            max_count: Maximum number of submissions to recalculate (None = all)

        Returns:
            List of recalculation results
        """
        submissions_needing_recalc = self.db.get_submissions_needing_recalc()
        if max_count:
            submissions_needing_recalc = submissions_needing_recalc[:max_count]

        results = []
        for submission_id in submissions_needing_recalc:
            result = self.recalculate_submission(submission_id, new_version)
            results.append(result)

        return results
