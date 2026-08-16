import { createHash } from 'node:crypto';
import {
  DOMAIN_REPOSITORIES,
  savepoint,
  type RepositoryTransactionStep
} from './repositories';
import type { RuntimeDatabase } from './runtimeDatabase';

const PROJECT_CONTEXT_ID_DOMAIN = 'limcode-reliable-project-context\0';
const CONVERSATION_PROJECT_LINK_ID_DOMAIN = 'limcode-reliable-conversation-project-link\0';

export interface ProjectFolderAssignment {
  uri: string;
  name: string;
}

/** Stable identity for one canonical workspace-folder URI. */
export function projectContextIdForUri(uriInput: string): string {
  const uri = requireText(uriInput, 'ProjectContext.uri');
  return `project_context_${digest(PROJECT_CONTEXT_ID_DOMAIN, uri)}`;
}

/** Stable one-primary-project relationship identity for a Conversation. */
export function conversationProjectLinkId(conversationIdInput: string): string {
  const conversationId = requireText(conversationIdInput, 'ConversationProjectLink.conversation_id');
  return `conversation_project_link_${digest(CONVERSATION_PROJECT_LINK_ID_DOMAIN, conversationId)}`;
}

/**
 * Prepare a canonical ProjectContext plus its independent Conversation relationship.
 *
 * The savepoint makes concurrent Extension Hosts converge on the same URI identity. The following
 * assert rejects a cryptographic-id collision or a conflicting canonical row instead of silently
 * binding the Conversation to unrelated project data.
 */
export function projectFolderAssignmentSteps(input: {
  conversationId: string;
  folder: ProjectFolderAssignment;
  now: string;
}): RepositoryTransactionStep[] {
  const conversationId = requireText(input.conversationId, 'conversationId');
  const uri = requireText(input.folder.uri, 'folder.uri');
  const name = requireText(input.folder.name, 'folder.name');
  const now = requireText(input.now, 'now');
  const projectContextId = projectContextIdForUri(uri);
  return [
    savepoint('project_context_identity', [
      DOMAIN_REPOSITORIES.domain('ProjectContext').insert({
        id: projectContextId,
        kind: 'folder',
        uri,
        name,
        created_at: now,
        updated_at: now
      })
    ], {
      kind: 'rollback-and-continue-on-unique',
      constraints: [
        { domain: 'ProjectContext', columns: ['id'] },
        { domain: 'ProjectContext', columns: ['uri'] }
      ]
    }),
    DOMAIN_REPOSITORIES.domain('ProjectContext').assert(projectContextId, {
      kind: 'folder',
      uri
    }),
    DOMAIN_REPOSITORIES.domain('ProjectContext').update(projectContextId, {
      name,
      updated_at: now
    }),
    conversationProjectLinkInsertStep({ conversationId, projectContextId, now })
  ];
}

/** 读取会话绑定的主项目；无绑定返回 undefined，多于一个绑定视为 Runtime 数据损坏。 */
export async function projectFolderForConversation(
  database: RuntimeDatabase,
  conversationIdInput: string
): Promise<ProjectFolderAssignment | undefined> {
  const conversationId = requireText(conversationIdInput, 'ConversationProjectLink.conversation_id');
  const linkSnapshot = await database.snapshot([
    DOMAIN_REPOSITORIES.domain('ConversationProjectLink').list({
      where: { conversation_id: conversationId, role: 'primary' },
      limit: 2
    })
  ]);
  const links = Array.isArray(linkSnapshot.snapshot[0]) ? linkSnapshot.snapshot[0] : [];
  if (links.length > 1) throw new Error(`Conversation ${conversationId} has multiple primary project links.`);
  const link = links[0];
  if (!link) return undefined;
  const projectContextId = requireText(link.project_context_id, 'ConversationProjectLink.project_context_id');
  const projectSnapshot = await database.snapshot([
    DOMAIN_REPOSITORIES.domain('ProjectContext').get(projectContextId)
  ]);
  const project = projectSnapshot.snapshot[0];
  if (!project || Array.isArray(project)) throw new Error(`ProjectContext ${projectContextId} does not exist.`);
  return {
    uri: requireText(project.uri, 'ProjectContext.uri'),
    name: requireText(project.name, 'ProjectContext.name')
  };
}

export function conversationProjectLinkInsertStep(input: {
  conversationId: string;
  projectContextId: string;
  now: string;
}): RepositoryTransactionStep {
  const conversationId = requireText(input.conversationId, 'conversationId');
  const projectContextId = requireText(input.projectContextId, 'projectContextId');
  const now = requireText(input.now, 'now');
  return DOMAIN_REPOSITORIES.domain('ConversationProjectLink').insert({
    id: conversationProjectLinkId(conversationId),
    conversation_id: conversationId,
    project_context_id: projectContextId,
    role: 'primary',
    created_at: now,
    updated_at: now
  });
}

function digest(domain: string, value: string): string {
  return createHash('sha256').update(domain).update(value).digest('hex');
}

function requireText(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${label} must be non-empty text.`);
  }
  return value.trim();
}
