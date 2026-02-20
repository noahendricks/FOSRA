from __future__ import annotations

from loguru import logger
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.src.api.schemas import (
    WorkspaceDeleteRequest,
    WorkspaceRequest,
    NewWorkspaceRequest,
    WorkspaceUpdateRequest,
)
from backend.src.domain.exceptions import (
    TenantContextError,
    WorkspaceAccessDenied,
    WorkspaceExistenceError,
    WorkspaceRetrievalError,
    WorkspaceStorageError,
)
from backend.src.domain.schemas import Workspace, WorkspaceFull
from backend.src.storage.utils.converters import orm_to_domain
from backend.src.storage.models import WorkspaceORM


class WorkspaceRepo:
    @staticmethod
    async def retrieve_workspace_by_id(
        workspace_request: WorkspaceRequest,
        session: AsyncSession,
    ) -> WorkspaceFull:
        try:
            query = select(WorkspaceORM).where(
                WorkspaceORM.user_id == workspace_request.user_id,
                WorkspaceORM.workspace_id == workspace_request.workspace_id,
            )

            result = await session.execute(query)
            workspace_orm = result.scalar_one_or_none()

            if not workspace_orm:
                raise WorkspaceExistenceError(
                    user_id=workspace_request.user_id,
                    workspace_id=workspace_request.workspace_id,
                    remediation="Verify that the workspace ID is correct and belongs to the user.",
                )
            return orm_to_domain(workspace_orm, WorkspaceFull)
        except Exception as e:
            logger.error(
                f"Error retrieving workspace {workspace_request.workspace_id}: {e}"
            )
            raise WorkspaceStorageError(
                user_id=workspace_request.user_id,
                workspace_id=workspace_request.workspace_id,
                operation="retrieve by workspace id",
                reason=str(e),
                remediation="Ensure the workspace ID is correct.",
            ) from e

    @staticmethod
    async def get_workspace_orm(
        session: AsyncSession, workspace_request: WorkspaceRequest
    ) -> WorkspaceORM:
        try:
            query = select(WorkspaceORM).where(
                WorkspaceORM.user_id == workspace_request.user_id,
                WorkspaceORM.workspace_id == workspace_request.workspace_id,
            )

            result = await session.execute(query)
            workspace_orm = result.scalar_one_or_none()

            if not workspace_orm:
                raise WorkspaceAccessDenied(
                    user_id=workspace_request.user_id,
                    workspace_id=workspace_request.workspace_id,
                    remediation="Verify that the workspace ID is correct and belongs to the user.",
                    resource_id=workspace_request.workspace_id,
                    resource_type="workspace",
                )

            return workspace_orm

        except Exception as e:
            logger.error(
                f"Error retrieving workspace ORM {workspace_request.workspace_id}: {e}"
            )
            raise WorkspaceRetrievalError(
                workspace_id=workspace_request.workspace_id,
                user_id=workspace_request.user_id,
                reason=str(e),
                remediation="Ensure the workspace ID is correct.",
            )

    @staticmethod
    async def get_all_workspaces(
        session: AsyncSession, user_id: str
    ) -> list[WorkspaceFull]:
        try:
            skip: int = 0
            limit: int = 100

            query = (
                select(WorkspaceORM)
                .where(WorkspaceORM.user_id == user_id)
                .offset(skip)
                .limit(limit)
                .order_by(WorkspaceORM.workspace_id)
            )

            result = await session.execute(query)
            workspaces = result.scalars().all()

            return [orm_to_domain(ws, WorkspaceFull) for ws in workspaces]
        except Exception as e:
            logger.error(f"Error retrieving workspaces for user {user_id}: {e}")
            raise WorkspaceRetrievalError(
                workspace_id="",
                user_id=user_id,
                reason=f"{str(e)}",
                remediation="Ensure the user ID is correct.",
            )

    @staticmethod
    async def create_workspace(
        request: NewWorkspaceRequest,
        session: AsyncSession,
    ) -> Workspace:
        try:
            if not request.name or not request.user_id:
                raise TenantContextError("name and user_id are required")

            workspace_orm: WorkspaceORM = WorkspaceORM(
                name=request.name,
                user_id=request.user_id,
            )

            session.add(workspace_orm)

            await session.commit()
            await session.refresh(workspace_orm)

            logger.debug(f"Created workspace {workspace_orm.workspace_id}")

            return orm_to_domain(workspace_orm, WorkspaceFull)

        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            raise WorkspaceStorageError(
                workspace_id="N/A",
                operation="create",
                user_id=request.user_id,
                reason=str(e),
            ) from e

    @staticmethod
    async def update_workspace(
        workspace_update: WorkspaceUpdateRequest,
        session: AsyncSession,
    ) -> WorkspaceFull:
        try:
            # WARN: Incomplete / Not Working
            query = select(WorkspaceORM).where(
                WorkspaceORM.workspace_id == workspace_update.workspace_id,
                WorkspaceORM.user_id == workspace_update.user_id,
            )

            result = await session.execute(query)
            workspace_orm = result.scalar_one_or_none()

            if not workspace_orm:
                raise WorkspaceExistenceError(
                    workspace_id=workspace_update.workspace_id,
                    user_id=workspace_update.user_id,
                    remediation="Verify that the workspace ID is correct and belongs to the user.",
                )

            # Apply updates
            update_dict = workspace_update.model_dump(exclude_unset=True)

            for key, value in update_dict.items():
                if hasattr(workspace_orm, key):
                    setattr(workspace_orm, key, value)

            await session.commit()
            await session.refresh(workspace_orm)

            logger.debug(f"Updated workspace {workspace_update.workspace_id}")
            return orm_to_domain(workspace_orm, WorkspaceFull)

        except Exception as e:
            logger.error(
                f"Error updating workspace {workspace_update.workspace_id}: {e}"
            )
            raise WorkspaceStorageError(
                operation="update",
                workspace_id=workspace_update.workspace_id,
                user_id=workspace_update.user_id,
                reason=str(e),
            ) from e

    @staticmethod
    async def delete_workspace(
        workspace_request: WorkspaceDeleteRequest,
        session: AsyncSession,
    ) -> bool:
        try:
            query = select(WorkspaceORM).where(
                WorkspaceORM.workspace_id == workspace_request.workspace_id,
                WorkspaceORM.user_id == workspace_request.user_id,
            )

            result = await session.execute(query)
            workspace_orm = result.scalar_one_or_none()

            if not workspace_orm:
                raise WorkspaceExistenceError(
                    workspace_id=workspace_request.workspace_id,
                    user_id=workspace_request.user_id,
                    remediation="Verify that the workspace ID is correct and belongs to the user.",
                )

            await session.delete(workspace_orm)
            await session.commit()

            logger.info(f"Deleted workspace {workspace_request.workspace_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error deleting workspace {workspace_request.workspace_id}: {e}"
            )
            raise WorkspaceStorageError(
                operation="delete",
                workspace_id=workspace_request.workspace_id,
                reason=str(e),
            )

    @staticmethod
    async def get_workspace_with_counts(
        workspace_request: WorkspaceRequest,
        session: AsyncSession,
    ) -> WorkspaceFull:
        """Get workspace with related entity counts.
        Returns:
            WorkspaceWithRelations with counts populated
        """

        try:
            query = (
                select(WorkspaceORM)
                .options(
                    selectinload(WorkspaceORM.sources),
                    selectinload(WorkspaceORM.convos),
                )
                .where(
                    WorkspaceORM.workspace_id == workspace_request.workspace_id,
                    WorkspaceORM.user_id == workspace_request.user_id,
                )
            )

            result = await session.execute(query)
            workspace_orm = result.scalar_one_or_none()

            if not workspace_orm:
                raise WorkspaceExistenceError(
                    workspace_id=workspace_request.workspace_id,
                    user_id=workspace_request.user_id,
                    remediation="Verify that the workspace ID is correct and belongs to the user.",
                )

            return WorkspaceFull(
                workspace_id=workspace_orm.workspace_id,
                user_id=workspace_orm.user_id,
                name=workspace_orm.name,
                description=workspace_orm.description,
                sources_count=len(workspace_orm.sources),
                conversations_count=len(workspace_orm.convos),
            )

        except Exception as e:
            logger.error(f"Error getting workspace with counts: {e}")
            raise WorkspaceRetrievalError(
                workspace_id=workspace_request.workspace_id,
                user_id=workspace_request.user_id,
                reason=str(e),
                remediation="Ensure the database is accessible.",
            )

    @staticmethod
    async def workspace_exists(
        workspace_request: WorkspaceRequest, session: AsyncSession
    ) -> bool:
        try:
            query = select(WorkspaceORM.workspace_id).where(
                WorkspaceORM.workspace_id == workspace_request.workspace_id,
                WorkspaceORM.user_id == workspace_request.user_id,
            )

            result = await session.execute(query)
            return result.scalar_one_or_none() is not None

        except Exception as e:
            logger.error(f"Error checking workspace existence: {e}")
            raise WorkspaceRetrievalError(
                workspace_id=workspace_request.workspace_id,
                user_id=workspace_request.user_id,
                reason=str(e),
                remediation="Ensure the database is accessible and the workspace ID is correct.",
            )

    @staticmethod
    async def get_or_create_default_workspace(
        workspace_request: WorkspaceRequest | NewWorkspaceRequest,
        session: AsyncSession,
    ) -> WorkspaceFull | Workspace:
        workspace: WorkspaceFull | Workspace | None = None

        if (
            isinstance(workspace_request, WorkspaceRequest)
            and workspace_request.workspace_id
        ):
            try:
                workspace = await WorkspaceRepo.retrieve_workspace_by_id(
                    session=session, workspace_request=workspace_request
                )
            except WorkspaceExistenceError:
                logger.info(f"Workspace {workspace_request.workspace_id} not found.")

        else:
            if isinstance(workspace_request, NewWorkspaceRequest):
                create_data = workspace_request
            else:
                create_data = NewWorkspaceRequest(
                    **workspace_request.model_dump(exclude={"workspace_id"}),
                    name="Default Workspace",
                    description="Automatically created default workspace.",
                )

            workspace: (
                WorkspaceFull | Workspace | None
            ) = await WorkspaceRepo.create_workspace(
                request=create_data, session=session
            )

        if not workspace:
            raise WorkspaceStorageError(
                operation="get or create default workspace",
                workspace_id="",
                reason="Failed to get or create default workspace.",
            )

        return workspace

    @staticmethod
    async def archive_convo(
        session: AsyncSession, workspace_id: str, convo_id: str, user_id: str
    ) -> list[str]:
        logger.info(f"Retrieving conversation: {convo_id}")

        try:
            # Get Workspace
            workspace: WorkspaceORM = await WorkspaceRepo.get_workspace_orm(
                session=session,
                workspace_request=WorkspaceRequest(
                    workspace_id=workspace_id, user_id=user_id
                ),
            )

            # Get archived convos list
            archived_convos: list[str] = workspace.archived_convos or []

            # Append convo id to list
            if convo_id not in archived_convos:
                archived_convos.append(convo_id)

                workspace.archived_convos = archived_convos

                session.add(workspace)

                await session.commit()
                await session.refresh(workspace)

            # Return updated workspace

            return archived_convos
        except Exception as e:
            raise e

    @staticmethod
    async def restore_convo(
        session: AsyncSession, workspace_id: str, convo_id: str, user_id: str
    ) -> list[str]:
        logger.info(f"Retrieving conversation: {convo_id}")

        try:
            # Get Workspace
            workspace: WorkspaceORM = await WorkspaceRepo.get_workspace_orm(
                session=session,
                workspace_request=WorkspaceRequest(
                    workspace_id=workspace_id, user_id=user_id
                ),
            )

            # Get archived convos list
            archived_convos: list[str] = workspace.archived_convos or []

            # Remove convo id from list
            if convo_id not in archived_convos:
                archived_convos.remove(convo_id)

                workspace.archived_convos = archived_convos

                session.add(workspace)

                await session.commit()
                await session.refresh(workspace)

            return archived_convos
        except Exception as e:
            raise e
