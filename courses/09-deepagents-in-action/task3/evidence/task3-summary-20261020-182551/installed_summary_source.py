class _DeepAgentsSummarizationMiddleware(AgentMiddleware):
    """Summarization middleware with backend for conversation history offloading."""

    state_schema = SummarizationState
    serialized_name: ClassVar[str] = "SummarizationMiddleware"
    """Preferred config-file reference for class-form exclusion export."""

    @property
    def name(self) -> str:
        """Report the public `SummarizationMiddleware` alias for string-form exclusion.

        The impl class is private (`_DeepAgentsSummarizationMiddleware`) but
        ships under the public `SummarizationMiddleware` name, so
        `excluded_middleware={"SummarizationMiddleware"}` targets this class.
        Subclasses fall back to `type(self).__name__` so user-authored
        extensions don't silently inherit the alias.
        """
        if type(self) is _DeepAgentsSummarizationMiddleware:
            return "SummarizationMiddleware"
        return type(self).__name__

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BackendProtocol,
        trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        truncate_args_settings: TruncateArgsSettings | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware with backend support.

        Args:
            model: The language model to use for generating summaries.
            backend: Backend instance for persisting conversation history.
            trigger: Threshold(s) that trigger summarization. A tuple is a single threshold,
                a dict combines thresholds with AND semantics, and a list combines items
                with OR semantics.
            keep: Context retention policy after summarization.

                Defaults to keeping last 20 messages.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Max tokens to include when generating summary.

                Defaults to 4000.
            truncate_args_settings: Settings for truncating large tool arguments in old messages.

                Provide a [`TruncateArgsSettings`][deepagents.middleware.summarization.TruncateArgsSettings]
                dictionary to configure when and how to truncate tool arguments. If `None`,
                argument truncation is disabled.

                !!! example

                    ```python
                    # Truncate when 50 messages is reached, ignoring the last 20 messages
                    {"trigger": ("messages", 50), "keep": ("messages", 20), "max_length": 2000, "truncation_text": "...(truncated)"}

                    # Truncate when 50% of context window reached, ignoring messages in last 10% of window
                    {"trigger": ("fraction", 0.5), "keep": ("fraction", 0.1), "max_length": 2000, "truncation_text": "...(truncated)"}

        Raises:
            TypeError: If the removed `history_path_prefix` argument is provided.

        Example:
            ```python
            from deepagents.middleware.summarization import SummarizationMiddleware
            from deepagents.backends import StateBackend

            middleware = SummarizationMiddleware(
                model="gpt-5.5",
                backend=StateBackend(),
                trigger=("tokens", 100000),
                keep=("messages", 20),
            )
            ```
        """
        if "history_path_prefix" in deprecated_kwargs:
            msg = "`history_path_prefix` was removed in deepagents 0.7. Configure `CompositeBackend.artifacts_root` instead."
            raise TypeError(msg)

        # Initialize langchain helper for core summarization logic
        self._lc_helper = LCSummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )

        # Whether the configured token counter accepts a `tools` kwarg. Resolved
        # once here (the counter is fixed after construction) so the per-call
        # token count never pays signature-introspection cost. `None` means the
        # signature could not be introspected, so `_count_tokens` probes instead.
        self._counter_accepts_tools = _token_counter_accepts_tools(self.token_counter)

        # Deep Agents specific attributes
        self._backend = backend

        artifacts_root = backend.artifacts_root if isinstance(backend, CompositeBackend) else "/"
        _root = artifacts_root.rstrip("/")
        self._history_path_prefix = f"{_root}/conversation_history"
        self._large_tool_results_prefix = f"{_root}/large_tool_results"

        self._media_prefix = f"{self._history_path_prefix}/media"

        # Parse truncate_args_settings
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep: ContextSize = ("messages", 20)
            self._max_arg_length = 2000
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 2000)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

    # Delegated properties and methods from langchain helper
    @property
    def model(self) -> BaseChatModel:
        """The language model used for generating summaries."""
        return self._lc_helper.model

    @property
    def token_counter(self) -> TokenCounter:
        """Function to count tokens in messages."""
        return self._lc_helper.token_counter

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        return self._lc_helper._get_profile_limits()

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        return self._lc_helper._should_summarize(messages, total_tokens)

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        return self._lc_helper._determine_cutoff_index(messages)

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        return self._lc_helper._partition_messages(conversation_messages, cutoff_index)

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages."""
        return self._lc_helper._create_summary(messages_to_summarize)

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages (async)."""
        return await self._lc_helper._acreate_summary(messages_to_summarize)

    def _get_session_id(self, state: Mapping[str, Any]) -> str:
        """Resolve the session id naming the offload history file.

        Reuses a previously persisted `_summarization_session_id` so history
        appends to one file across turns; otherwise generates a fresh id, which
        the caller persists in the state update so later turns reuse it.

        The id is internal and scoped per graph invocation, so each invocation
        -- including each sub-agent -- gets its own history file.

        Args:
            state: The agent state to read the persisted id from.

        Returns:
            A session id (e.g. `'session_<uuid4 hex>'`).
        """
        existing = state.get("_summarization_session_id")
        if isinstance(existing, str) and existing:
            return existing
        # Full uuid4 entropy: history filenames must not collide across
        # independent sessions sharing a backend, or their evicted history mixes.
        return f"session_{uuid.uuid4().hex}"

    def _get_history_path(self, session_id: str) -> str:
        """Generate path for storing conversation history.

        Returns a single file per session id that gets appended to over time.

        Args:
            session_id: An id from `_get_session_id`.

        Returns:
            Path string like `'/conversation_history/{session_id}.md'`
        """
        return f"{self._history_path_prefix}/{session_id}.md"

    def _is_summary_message(self, msg: AnyMessage) -> bool:
        """Check if a message is a previous summarization message.

        Summary messages are `HumanMessage` objects with `lc_source='summarization'` in
        `additional_kwargs`. These should be filtered from offloads to avoid redundant
        storage during chained summarization.

        Args:
            msg: Message to check.

        Returns:
            Whether this is a summary `HumanMessage` from a previous summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the previous
        summary `HumanMessage` since the original messages are already stored in the
        backend.

        Args:
            messages: List of messages to filter.

        Returns:
            Messages without previous summary `HumanMessage` objects.
        """
        return [msg for msg in messages if not self._is_summary_message(msg)]

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The generated summary text.
            file_path: Path where conversation history was stored, or `None`.

                Optional since offloading may fail.

        Returns:
            List containing the summary `HumanMessage`.
        """
        if file_path is not None:
            content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        return [
            HumanMessage(
                content=content,
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _get_effective_messages(self, request: ModelRequest) -> list[AnyMessage]:
        """Generate effective messages for model call based on summarization event.

        Delegates to `_apply_event_to_messages` so the defensive checks
        (malformed event, out-of-bounds cutoff) are shared with the compact
        tool path.

        Args:
            request: The model request with messages from state.

        Returns:
            The effective message list to use for the model call.
        """
        event = request.state.get("_summarization_event")
        return self._apply_event_to_messages(request.messages, event)

    @staticmethod
    def _apply_event_to_messages(
        messages: list[AnyMessage],
        event: SummarizationEvent | None,
    ) -> list[AnyMessage]:
        """Reconstruct effective messages from raw state messages and a summarization event.

        When a prior summarization event exists, the effective conversation is
        the summary message followed by all messages from `cutoff_index` onward.

        Args:
            messages: Full message list from state.
            event: The `_summarization_event` dict, or `None`.

        Returns:
            The effective message list the model would see.
        """
        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed _summarization_event (missing keys): %s", exc)
            return list(messages)

        if cutoff_idx > len(messages):
            logger.warning(
                "Summarization cutoff_index %d exceeds message count %d; remaining slice will be empty",
                cutoff_idx,
                len(messages),
            )
            return [summary_msg]

        result: list[AnyMessage] = [summary_msg]
        result.extend(messages[cutoff_idx:])
        return result

    @staticmethod
    def _compute_state_cutoff(
        event: SummarizationEvent | None,
        effective_cutoff: int,
    ) -> int:
        """Translate an effective-list cutoff index to an absolute state index.

        When a prior summarization event exists, the effective message list
        starts with the summary message at index 0. The -1 accounts for the
        summary message at effective index 0, which does not correspond to a
        real state message -- the effective cutoff already counts it, so we
        subtract 1 to avoid double-counting.

        Args:
            event: The prior `_summarization_event`, or `None`.
            effective_cutoff: Cutoff index within the effective message list.

        Returns:
            The absolute cutoff index for the state.
        """
        if event is None:
            return effective_cutoff
        prior_cutoff = event.get("cutoff_index")
        if not isinstance(prior_cutoff, int):
            logger.warning("Malformed _summarization_event: missing cutoff_index")
            return effective_cutoff
        return prior_cutoff + effective_cutoff - 1

    def _should_truncate_args(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Check if argument truncation should be triggered.

        Args:
            messages: Current message history.
            total_tokens: Total token count of messages.

        Returns:
            True if truncation should occur, False otherwise.
        """
        if self._truncate_args_trigger is None:
            return False

        trigger_type, trigger_value = self._truncate_args_trigger

        if trigger_type == "messages":
            return len(messages) >= trigger_value
        if trigger_type == "tokens":
            return total_tokens >= trigger_value
        if trigger_type == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * trigger_value)
            if threshold <= 0:
                threshold = 1
            return total_tokens >= threshold

        return False

    def _determine_truncate_cutoff_index(self, messages: list[AnyMessage]) -> int:  # noqa: PLR0911
        """Determine the cutoff index for argument truncation based on keep policy.

        Messages at index >= cutoff should be preserved without truncation.
        Messages at index < cutoff can have their tool args truncated.

        Args:
            messages: Current message history.

        Returns:
            Index where truncation cutoff occurs. Messages before this index
            should have args truncated, messages at/after should be preserved.
        """
        keep_type, keep_value = self._truncate_args_keep

        if keep_type == "messages":
            # Keep the most recent N messages
            if len(messages) <= keep_value:
                return len(messages)  # All messages are recent
            return int(len(messages) - keep_value)

        if keep_type in {"tokens", "fraction"}:
            # Calculate target token count
            if keep_type == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None:
                    # Fallback to message count if profile not available
                    messages_to_keep = 20
                    if len(messages) <= messages_to_keep:
                        return len(messages)
                    return len(messages) - messages_to_keep
                target_token_count = int(max_input_tokens * keep_value)
            else:
                target_token_count = int(keep_value)

            if target_token_count <= 0:
                target_token_count = 1

            # Keep recent messages up to token limit
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = self._lc_helper._partial_token_counter([messages[i]])
                if tokens_kept + msg_tokens > target_token_count:
                    return i + 1
                tokens_kept += msg_tokens
            return 0  # All messages are within token limit

        return len(messages)

    def _truncate_tool_call(self, tool_call: ToolCall) -> ToolCall:
        """Truncate large arguments in a single tool call.

        Args:
            tool_call: The tool call dictionary to truncate.

        Returns:
            A copy of the tool call with large arguments truncated.
        """
        args = tool_call.get("args", {})

        truncated_args = {}
        modified = False

        for key, value in args.items():
            if isinstance(value, str) and len(value) > self._max_arg_length:
                truncated_args[key] = value[:20] + self._truncation_text
                modified = True
            else:
                truncated_args[key] = value

        if modified:
            return {
                **tool_call,
                "args": truncated_args,
            }
        return tool_call

    def _count_tokens(
        self,
        messages: list[AnyMessage],
        system_message: SystemMessage | None,
        tools: list[BaseTool | dict[str, Any]] | None,
    ) -> int:
        """Count tokens for messages plus optional system message and tools.

        Args:
            messages: Messages to count.
            system_message: Optional system message prepended before counting.
            tools: Optional tools whose schemas contribute to the count.

        Returns:
            Total token count. Counts without `tools` when the configured
                `token_counter` does not accept a `tools` keyword. When the
                counter's signature is introspectable, a `TypeError` raised
                inside the counter's own body is never masked — it propagates so
                a broken counter is not hidden behind a silently wrong count.
                Counters whose signature cannot be introspected are probed
                instead, and only there does a `TypeError` fall back to counting
                without `tools`.
        """
        counted_messages = [system_message, *messages] if system_message is not None else messages
        if self._counter_accepts_tools is True:
            # `tools=` is absent from the `TokenCounter` protocol but accepted
            # here: the signature check above confirmed the counter takes it.
            return self.token_counter(counted_messages, tools=tools)  # ty: ignore[unknown-argument]
        if self._counter_accepts_tools is False:
            return self.token_counter(counted_messages)
        # Signature could not be introspected; probe defensively. This is the
        # only path that swallows a `TypeError`, and only for counters whose
        # signature is opaque (some C-level callables expose no signature).
        try:
            # `tools=` is outside the `TokenCounter` protocol; the probe verifies
            # acceptance at runtime, falling back below if it is rejected.
            return self.token_counter(counted_messages, tools=tools)  # ty: ignore[unknown-argument]
        except TypeError:
            return self.token_counter(counted_messages)

    def _truncate_args(
        self,
        messages: list[AnyMessage],
        total_tokens: int,
    ) -> tuple[list[AnyMessage], bool]:
        """Truncate large tool call arguments in old messages.

        Args:
            messages: Messages to potentially truncate.
            total_tokens: Precomputed token count for `messages` (plus system
                message and tools). Counting tools is expensive (schema
                conversion per tool), so the caller counts once and shares the
                result across the truncation and summarization checks.

        Returns:
            Tuple of (truncated_messages, modified). If modified is False,
            truncated_messages is the same as input messages.
        """
        if not self._should_truncate_args(messages, total_tokens):
            return messages, False

        cutoff_index = self._determine_truncate_cutoff_index(messages)
        if cutoff_index >= len(messages):
            return messages, False

        # Process messages before the cutoff
        truncated_messages = []
        modified = False

        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
                # Check if this AIMessage has tool calls we need to truncate
                truncated_tool_calls = []
                msg_modified = False

                for tool_call in msg.tool_calls:
                    if tool_call["name"] in {"write_file", "edit_file"}:
                        truncated_call = self._truncate_tool_call(tool_call)
                        if truncated_call != tool_call:
                            msg_modified = True
                        truncated_tool_calls.append(truncated_call)
                    else:
                        truncated_tool_calls.append(tool_call)

                if msg_modified:
                    # Create a new AIMessage with truncated tool calls
                    truncated_msg = msg.model_copy()
                    truncated_msg.tool_calls = truncated_tool_calls
                    truncated_messages.append(truncated_msg)
                    modified = True
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)

        return truncated_messages, modified

    def _offload_inline_media(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], int]:
        """Decode inline `data:` media blocks to files and replace them with path references.

        Covers any inline `data:` URL (base64 or percent-encoded/plaintext), not
        just base64, because the XML history renderer drops every inline `data:`
        URL. The caller uploads media before both `_offload_to_backend` and
        `_create_summary`, so both paths receive messages with inline data
        replaced by path references (or error placeholders when an upload fails).
        The archive keeps addressable `<image url="..." />` references, and the
        summary prompt does not receive raw media bytes.

        Each unique media file is uploaded once to
        `{artifacts_root}/conversation_history/media/{sha256[:16]}.{ext}` (the
        prefix follows the backend's `artifacts_root`, defaulting to `/`).
        Identical media across messages are deduped by content hash.

        Failures are tracked per block. A block whose upload failed -- or whose
        payload could not be decoded -- is replaced with an
        `<image error="failed_to_offload" />` text placeholder; a successfully
        uploaded block is rewritten to a typed media reference block. The caller
        receives the count of failed blocks so it can warn that those media are
        unrecoverable.

        Args:
            backend: Backend to write media files to.
            messages: Messages to process.

        Returns:
            A `(messages, failed_block_count)` tuple. `messages` has base64
                blocks replaced by path-reference media blocks or error
                placeholders; messages without base64 content are returned
                unchanged. `failed_block_count` is the number of media blocks
                that became failed-offload placeholders.
        """
        path_map: dict[str, str] = {}  # key -> backend path (successfully uploaded)
        failed_keys: set[str] = set()  # keys whose upload failed
        saw_inline_media = False

        # First pass: upload each unique media file individually for per-block failure tracking.
        for msg in messages:
            for block in msg.content_blocks:
                data_url = _extract_data_url(block)
                if data_url is None:
                    continue
                saw_inline_media = True
                decoded = _decode_data_url(data_url)
                if decoded is None:
                    continue  # undecodable; rewrite emits a failed-offload placeholder
                raw, ext, _mime = decoded
                key = hashlib.sha256(raw).hexdigest()[:16]
                if key in path_map or key in failed_keys:
                    continue
                img_path = f"{self._media_prefix}/{key}.{ext}"
                try:
                    responses = backend.upload_files([(img_path, raw)])
                    if error := _upload_response_error(responses):
                        logger.warning(
                            "Failed to upload media %s to backend: %s",
                            img_path,
                            error,
                        )
                        failed_keys.add(key)
                        continue
                    path_map[key] = img_path
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to upload media %s to backend: %s: %s",
                        img_path,
                        type(e).__name__,
                        e,
                    )
                    failed_keys.add(key)

        if not saw_inline_media:
            return messages, 0  # no inline media present; return originals unchanged

        return _rewrite_data_url_blocks(messages, path_map)

    async def _aoffload_inline_media(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], int]:
        """Async twin of `_offload_inline_media` using `aupload_files`.

        See `_offload_inline_media` for full documentation, including the
        `(messages, failed_block_count)` return contract.
        """
        path_map: dict[str, str] = {}
        failed_keys: set[str] = set()
        saw_inline_media = False

        for msg in messages:
            for block in msg.content_blocks:
                data_url = _extract_data_url(block)
                if data_url is None:
                    continue
                saw_inline_media = True
                decoded = _decode_data_url(data_url)
                if decoded is None:
                    continue  # undecodable; rewrite emits a failed-offload placeholder
                raw, ext, _mime = decoded
                key = hashlib.sha256(raw).hexdigest()[:16]
                if key in path_map or key in failed_keys:
                    continue
                img_path = f"{self._media_prefix}/{key}.{ext}"
                try:
                    responses = await backend.aupload_files([(img_path, raw)])
                    if error := _upload_response_error(responses):
                        logger.warning(
                            "Failed to upload media %s to backend: %s",
                            img_path,
                            error,
                        )
                        failed_keys.add(key)
                        continue
                    path_map[key] = img_path
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to upload media %s to backend: %s: %s",
                        img_path,
                        type(e).__name__,
                        e,
                    )
                    failed_keys.add(key)

        if not saw_inline_media:
            return messages, 0

        return _rewrite_data_url_blocks(messages, path_map)

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        session_id: str,
    ) -> str | None:
        """Persist messages to backend before summarization.

        Appends evicted messages to a single markdown file per session. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        A `None` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            session_id: Id naming the history file.

        Returns:
            The file path where history was offloaded, or `None` on failure.
        """
        path = self._get_history_path(session_id)

        # Filter out previous summary messages to avoid redundant storage.
        # Base64 images are already converted to path references by the caller.
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages, format='xml')}\n\n"

        # Read existing content (if any) and append.
        # Note: We use download_files() instead of read() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = backend.edit(path, existing_content, combined_content) if existing_content else backend.write(path, combined_content)
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    async def _aoffload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        session_id: str,
    ) -> str | None:
        """Persist messages to backend before summarization (async).

        Appends evicted messages to a single markdown file per session. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        A `None` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            session_id: Id naming the history file.

        Returns:
            The file path where history was offloaded, or `None` on failure.
        """
        path = self._get_history_path(session_id)

        # Filter out previous summary messages to avoid redundant storage.
        # Base64 images are already converted to path references by the caller.
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages, format='xml')}\n\n"

        # Read existing content (if any) and append.
        # Note: We use adownload_files() instead of aread() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = await backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = (
                await backend.aedit(path, existing_content, combined_content) if existing_content else await backend.awrite(path, combined_content)
            )
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation.

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Control flow details:

        - If thresholds say "do not summarize", we still attempt one normal
            model call with the current effective/truncated messages.
        - If that call raises `ContextOverflowError`, we immediately fall back to
            the summarization path and retry the model call with
            `summary_message + preserved_recent_messages`.

        Unlike the legacy `before_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            A plain `ModelResponse` when no summarization event is created, or
                an `ExtendedModelResponse` that updates `_summarization_event`
                with `cutoff_index`, `summary_message`, and `file_path`.

                If `cutoff_index <= 0`, no compaction occurs and no
                `_summarization_event` update is emitted.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Count once; tool-schema conversion makes each count expensive, so the
        # count is shared between the truncation check and the summarize check.
        total_tokens = self._count_tokens(effective_messages, request.system_message, request.tools)

        # Step 1: Truncate args if configured
        truncated_messages, truncate_modified = self._truncate_args(
            effective_messages,
            total_tokens,
        )

        # Step 2: Check if summarization should happen
        if truncate_modified:
            total_tokens = self._count_tokens(truncated_messages, request.system_message, request.tools)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        overflow_triggered = False
        if not should_summarize:
            try:
                return handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                overflow_triggered = True
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        backend = self._backend
        # On overflow, offload the large preserved tail TM batch to per-TM files.
        new_state_tail: list[AnyMessage] = []
        if overflow_triggered:
            preserved_messages, new_state_tail = _clip_overflow_tail(
                preserved_messages,
                backend,
                keep=self._lc_helper.keep,
                max_input_tokens=self._get_profile_limits(),
                token_counter=self.token_counter,
                large_tool_results_prefix=self._large_tool_results_prefix,
            )

        # Upload inline media once so both offload and summary see path references.
        offloaded_media_messages, failed_media = self._offload_inline_media(backend, messages_to_summarize)

        # Resolve the internal history-file id and persist it below so later turns
        # append to the same file.
        session_id = self._get_session_id(request.state)

        # Offload to backend first so history is preserved before summarization.
        # If offload fails, summarization still proceeds (with file_path=None).
        file_path = self._offload_to_backend(backend, offloaded_media_messages, session_id)
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)
        elif failed_media:
            # History was saved, but some media became failed-offload placeholders.
            # Tie the warning to the saved file so the recovery pointer is honest.
            msg = (
                f"Conversation history was offloaded to {file_path}, but {failed_media} media "
                "block(s) could not be offloaded and appear as failed placeholders in the saved "
                "history; the original media is not recoverable."
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        # Generate summary
        summary = self._create_summary(offloaded_media_messages)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = handler(request.override(messages=modified_messages))

        update: dict[str, Any] = {
            "_summarization_event": new_event,
            "_summarization_session_id": session_id,
        }
        if new_state_tail:
            update["messages"] = list(new_state_tail)

        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update=update),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation (async).

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Control flow details:

        - If thresholds say "do not summarize", we still attempt one normal
            model call with the current effective/truncated messages.
        - If that call raises `ContextOverflowError`, we immediately fall back
            to the summarization path and retry the model call with
            `summary_message + preserved_recent_messages`.

        Unlike the legacy `abefore_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            A plain `ModelResponse` when no summarization event is created, or
                an `ExtendedModelResponse` that updates `_summarization_event`
                with `cutoff_index`, `summary_message`, and `file_path`.

                If `cutoff_index <= 0`, no compaction occurs and no
                `_summarization_event` update is emitted.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Count once; tool-schema conversion makes each count expensive, so the
        # count is shared between the truncation check and the summarize check.
        total_tokens = self._count_tokens(effective_messages, request.system_message, request.tools)

        # Step 1: Truncate args if configured
        truncated_messages, truncate_modified = self._truncate_args(
            effective_messages,
            total_tokens,
        )

        # Step 2: Check if summarization should happen
        if truncate_modified:
            total_tokens = self._count_tokens(truncated_messages, request.system_message, request.tools)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        overflow_triggered = False
        if not should_summarize:
            try:
                return await handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                overflow_triggered = True
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return await handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        backend = self._backend
        # On overflow, offload the large preserved tail TM batch to per-TM files.
        new_state_tail: list[AnyMessage] = []
        if overflow_triggered:
            preserved_messages, new_state_tail = await _aclip_overflow_tail(
                preserved_messages,
                backend,
                keep=self._lc_helper.keep,
                max_input_tokens=self._get_profile_limits(),
                token_counter=self.token_counter,
                large_tool_results_prefix=self._large_tool_results_prefix,
            )

        # Upload inline media once so both offload and summary see path references.
        # This must complete before the gather since both methods consume the result.
        offloaded_media_messages, failed_media = await self._aoffload_inline_media(backend, messages_to_summarize)

        # Resolve the internal history-file id and persist it below so later turns
        # append to the same file.
        session_id = self._get_session_id(request.state)

        # Offload to backend and generate summary concurrently -- they are independent.
        # If offload fails, summarization still proceeds (with file_path=None).
        file_path, summary = await asyncio.gather(
            self._aoffload_to_backend(backend, offloaded_media_messages, session_id),
            self._acreate_summary(offloaded_media_messages),
        )
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)
        elif failed_media:
            # History was saved, but some media became failed-offload placeholders.
            # Tie the warning to the saved file so the recovery pointer is honest.
            msg = (
                f"Conversation history was offloaded to {file_path}, but {failed_media} media "
                "block(s) could not be offloaded and appear as failed placeholders in the saved "
                "history; the original media is not recoverable."
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = await handler(request.override(messages=modified_messages))

        update: dict[str, Any] = {
            "_summarization_event": new_event,
            "_summarization_session_id": session_id,
        }
        if new_state_tail:
            update["messages"] = list(new_state_tail)

        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update=update),
        )
