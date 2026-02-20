from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas import (
    ConvoRequest,
    NewWorkspaceRequest,
    WorkspaceDeleteRequest,
    WorkspaceFullResponse,
    WorkspaceRequest,
    WorkspaceUpdateRequest,
)

from backend.src.api.schemas.api_schemas import NewWorkspaceResponse
from backend.src.domain.exceptions import (
    WorkspaceRetrievalError,
)
from backend.src.storage.workspace import WorkspaceRepo
from backend.src.storage.utils.converters import domain_to_response


class WorkspaceService:
    @staticmethod
    async def get_or_create_default_workspace(
        workspace_request: WorkspaceRequest | NewWorkspaceRequest,
        session: AsyncSession,
    ) -> WorkspaceFullResponse:
        try:
            workspace = await WorkspaceRepo().get_or_create_default_workspace(
                workspace_request=workspace_request, session=session
            )

            return domain_to_response(workspace, WorkspaceFullResponse)
        except Exception:
            raise

    @staticmethod
    async def retrieve_workspace_by_id(
        workspace_request: WorkspaceRequest, session: AsyncSession
    ) -> WorkspaceFullResponse:
        try:
            workspace = await WorkspaceRepo().retrieve_workspace_by_id(
                session=session, workspace_request=workspace_request
            )
            return domain_to_response(workspace, WorkspaceFullResponse)
        except Exception as e:
            logger.error(
                f"Error retrieving workspace {workspace_request.workspace_id}: {e}"
            )
            raise WorkspaceRetrievalError(
                workspace_id=workspace_request.workspace_id,
                reason=f"Failed to retrieve workspace: {e}",
                remediation="Verify the workspace ID is correct and exists.",
            ) from e

    @staticmethod
    async def get_all_workspaces(
        session: AsyncSession, user_id: str
    ) -> list[WorkspaceFullResponse]:
        workspaces = await WorkspaceRepo.get_all_workspaces(
            session=session, user_id=user_id
        )
        return [domain_to_response(ws, WorkspaceFullResponse) for ws in workspaces]

    @staticmethod
    async def create_workspace(
        create_workspace: NewWorkspaceRequest,
        session: AsyncSession,
    ) -> NewWorkspaceResponse:
        workspace = await WorkspaceRepo.create_workspace(
            request=create_workspace, session=session
        )
        logger.info(
            f"Created workspace {workspace.workspace_id} for user {workspace.user_id}"
        )
        return domain_to_response(workspace, NewWorkspaceResponse)

    @staticmethod
    async def update_workspace(
        workspace_update: WorkspaceUpdateRequest,
        session: AsyncSession,
    ) -> WorkspaceFullResponse:
        workspace = await WorkspaceRepo.update_workspace(
            workspace_update=workspace_update, session=session
        )
        logger.info(
            f"Updated workspace {workspace.workspace_id} for user {workspace.user_id}"
        )
        return domain_to_response(workspace, WorkspaceFullResponse)

    @staticmethod
    async def delete_workspace(
        workspace_request: WorkspaceDeleteRequest,
        session: AsyncSession,
    ) -> bool:
        result = await WorkspaceRepo.delete_workspace(
            workspace_request=workspace_request, session=session
        )
        logger.info(
            f"Deleted workspace {workspace_request.workspace_id} for user {
                workspace_request.user_id
            }"
        )
        return result

    @staticmethod
    async def delete_list_of_workspaces(
        workspace_request: WorkspaceDeleteRequest,
        session: AsyncSession,
    ) -> bool:
        result = False

        try:
            if workspace_request.workspace_list:
                ws_list = workspace_request.workspace_list
                for ws in ws_list:
                    result = await WorkspaceRepo.delete_workspace(
                        workspace_request=workspace_request, session=session
                    )

                    logger.info(
                        f"Deleted workspace {ws} for user: {workspace_request.user_id}"
                    )
            return result

        except Exception:
            raise

    @staticmethod
    async def get_workspace_with_counts(
        workspace_request: WorkspaceRequest,
        session: AsyncSession,
    ) -> WorkspaceFullResponse:
        workspace = await WorkspaceRepo.get_workspace_with_counts(
            workspace_request=workspace_request, session=session
        )
        return domain_to_response(workspace, WorkspaceFullResponse)

    @staticmethod
    async def workspace_exists(
        workspace_request: WorkspaceRequest, session: AsyncSession
    ) -> bool:
        exists = await WorkspaceRepo.workspace_exists(
            workspace_request=workspace_request, session=session
        )
        return exists

    @staticmethod
    async def archive_convo(
        convo_request: ConvoRequest, session: AsyncSession
    ) -> list[str]:
        archive = await WorkspaceRepo.archive_convo(
            workspace_id=convo_request.workspace_id,
            user_id=convo_request.user_id,
            convo_id=convo_request.convo_id,
            session=session,
        )
        return archive

    @staticmethod
    async def restore_convo(
        convo_request: ConvoRequest, session: AsyncSession
    ) -> list[str]:
        archive = await WorkspaceRepo.restore_convo(
            workspace_id=convo_request.workspace_id,
            user_id=convo_request.user_id,
            convo_id=convo_request.convo_id,
            session=session,
        )
        return archive
