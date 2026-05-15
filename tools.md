# Tools to Explore

Deferred tools available via `ToolSearch` in Claude Code sessions. Schemas are not loaded until requested — call `ToolSearch` with `select:<tool_name>` to fetch a tool's schema before invoking it.

## Discovery via `ToolSearch`

Claude Code uses a **deferred-registration** pattern: only the names of available tools are in context at session start, not their JSON schemas. Loading every schema eagerly would cost 10–20k+ tokens for the browser MCP alone, so the harness exposes a single meta-tool — `ToolSearch` — that fetches schemas on demand.

### How it works

1. At session start, deferred tool **names** are listed in a `<system-reminder>` (cheap — ~few hundred tokens).
2. Calling a deferred tool directly fails with `InputValidationError` because its parameter schema isn't loaded.
3. `ToolSearch` takes a query, matches it against the deferred list, and returns matched tools' full `<function>` definitions inline. Once a definition appears in the result, the tool is callable like any natively registered tool for the rest of the session.

### Query forms

- `select:<name>[,<name>...]` — fetch exact tools by name. Use when you know what you want.
  - Example: `select:WebFetch,WebSearch`
- `<keywords>` — fuzzy keyword search, ranked by relevance. Returns up to `max_results` matches.
  - Example: `notebook jupyter` → matches `NotebookEdit`
- `+<term> <other terms>` — require the `+`-prefixed term in the tool name, then rank by remaining terms.
  - Example: `+slack send` → prefers `slack_send_message` over `slack_read_thread`

### When to use

- **One-off task** → `select:` the exact tool(s) and call.
- **Exploring** ("is there a tool for X?") → keyword search to see candidates.
- **MCP tools** (`mcp__*`) **must** be loaded via `ToolSearch` before first use — the MCP server's own instructions enforce this.

### Tradeoffs

- **Saves context**: ~64 tool names ≈ a few hundred tokens vs. full schemas ≈ 10–20k+ tokens.
- **Costs a round-trip**: each new tool needs one `ToolSearch` call before the first invocation.
- **Names are the only map**: the model can only request tools whose existence it remembers from the system reminder — if a tool isn't named in context, it's invisible. Curate the list.

This is the same pattern the harness's own `--tool-set` profiles solve statically: pick a curated bundle (`core` / `coding` / `memory` / `diagnostic` / `research` / `ops` / `full`) targeting ~1.5k tokens of schema overhead, then accept the model can't reach outside the bundle without a restart. Dynamic (`ToolSearch`) and static (`--tool-set`) are different points on the same curve.

## Claude Code primitives

- `CronCreate` — schedule a recurring agent run
- `CronDelete` — remove a scheduled agent
- `CronList` — list scheduled agents
- `EnterPlanMode` — switch to plan-only mode (no edits)
- `EnterWorktree` — open an isolated git worktree
- `ExitPlanMode` — leave plan mode, present plan for approval
- `ExitWorktree` — close worktree, surface branch/path
- `LSP` — language-server protocol queries (definitions, references, diagnostics)
- `ListMcpResourcesTool` — enumerate resources exposed by connected MCP servers
- `Monitor` — stream stdout of a background process until a condition is met
- `NotebookEdit` — edit Jupyter `.ipynb` cells
- `PushNotification` — send a desktop push notification
- `ReadMcpResourceTool` — fetch a specific MCP resource by URI
- `RemoteTrigger` — kick off a remote agent run
- `SendMessage` — resume a running agent by name/ID with full context
- `TaskCreate` — create a tracked task (harness-managed, separate from beads)
- `TaskGet` — read a single task by ID
- `TaskList` — list tracked tasks
- `TaskOutput` — fetch a task's accumulated output
- `TaskStop` — stop a running task
- `TaskUpdate` — update a task's status/fields
- `TeamCreate` — create a team for agent spawning
- `TeamDelete` — delete a team
- `WebFetch` — fetch a URL, return rendered text
- `WebSearch` — keyword web search

## Chrome browser MCP (`mcp__claude-in-chrome__*`)

Requires loading before first use. Always call `tabs_context_mcp` first to discover existing tabs; never reuse tab IDs across sessions.

- `browser_batch` — batch multiple browser actions in one call
- `computer` — pixel-level click/type/screenshot (computer-use style)
- `file_upload` — upload a file to a file input
- `find` — locate an element on the page
- `form_input` — fill form fields
- `get_page_text` — read visible page text
- `gif_creator` — record an animated GIF of browser actions
- `javascript_tool` — run arbitrary JS in the page context
- `list_connected_browsers` — enumerate connected Chrome instances
- `navigate` — go to a URL
- `read_console_messages` — read JS console output (supports regex filter)
- `read_network_requests` — read network activity
- `read_page` — structured page read (DOM-aware)
- `resize_window` — change browser window size
- `select_browser` — choose which connected browser to target
- `shortcuts_execute` — run a named keyboard shortcut
- `shortcuts_list` — list available shortcuts
- `switch_browser` — switch active browser
- `tabs_close_mcp` — close a tab
- `tabs_context_mcp` — list current tabs (call first each session)
- `tabs_create_mcp` — open a new tab
- `upload_image` — upload an image asset

## Gmail MCP (`mcp__claude_ai_Gmail__*`)

- `authenticate` — start OAuth flow
- `complete_authentication` — finish OAuth flow

## Google Drive MCP (`mcp__claude_ai_Google_Drive__*`)

- `authenticate` — start OAuth flow
- `complete_authentication` — finish OAuth flow

## Slack MCP (`mcp__claude_ai_Slack__*`)

- `slack_create_canvas` — create a Slack canvas doc
- `slack_read_canvas` — read a canvas
- `slack_read_channel` — read channel messages
- `slack_read_thread` — read a thread
- `slack_read_user_profile` — fetch a user's profile
- `slack_schedule_message` — schedule a message for future send
- `slack_search_channels` — search for channels
- `slack_search_public` — search public channels
- `slack_search_public_and_private` — search both public and private channels
- `slack_search_users` — search for users
- `slack_send_message` — send a message
- `slack_send_message_draft` — draft a message without sending
- `slack_update_canvas` — edit an existing canvas
