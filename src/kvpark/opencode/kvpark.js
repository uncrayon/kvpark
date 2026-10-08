// kvpark-opencode-plugin 1
// kvpark for OpenCode. `kvpark setup --opencode` installs this file in an OpenCode plugins
// directory and `kvpark uninstall --opencode` removes it; it has no dependencies.
//
// It tags each chat request that OpenCode sends to the "kvpark" provider, so the kvpark proxy
// knows which conversation the request belongs to and can park and resume that conversation's
// KV cache (docs/opencode.md):
//   slot_archive_role      "foreground" for an agent working in a top-level session;
//                          "background" for session titles, summaries, compaction and subagents
//                          (they run in child sessions), which must not take over a conversation
//   slot_archive_key       "opencode:<session id>", on foreground requests only
//   slot_archive_autosave  true on foreground requests: save this conversation to disk before
//                          other work takes the model server's only slot
// The proxy removes these fields before the request reaches the model server. Requests to other
// providers are left unchanged.

const PROVIDER = "kvpark"
const NAMESPACE = "opencode:"
// OpenCode's hidden agents; they share the session of the conversation they serve.
const BACKGROUND_AGENTS = new Set(["title", "summary", "compaction"])

export const KvparkPlugin = async ({ client, directory }) => {
  // session id -> whether it is a top-level session (a subagent's session has a parent)
  const topLevel = new Map()

  async function isTopLevel(sessionID) {
    if (topLevel.has(sessionID)) return topLevel.get(sessionID)
    let result
    try {
      const response = await client.session.get({ path: { id: sessionID }, query: { directory } })
      if (!response?.data) return false // not verified: send it as background work
      result = !response.data.parentID
    } catch {
      return false
    }
    if (topLevel.size >= 1000) topLevel.delete(topLevel.keys().next().value)
    topLevel.set(sessionID, result)
    return result
  }

  return {
    "chat.params": async (input, output) => {
      if (input.model?.providerID !== PROVIDER || !input.sessionID) return
      const foreground = !BACKGROUND_AGENTS.has(input.agent) && (await isTopLevel(input.sessionID))
      output.options = output.options ?? {}
      output.options.slot_archive_role = foreground ? "foreground" : "background"
      if (foreground) {
        output.options.slot_archive_key = NAMESPACE + input.sessionID
        output.options.slot_archive_autosave = true
      }
    },
  }
}
