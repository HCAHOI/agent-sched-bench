def extract(query):
    command = query.get("current_command")
    if not isinstance(command, str) or not re.fullmatch(r"\s*(?:python3?\s+-m\s+pytest|pytest)(?:\s+-[A-Za-z0-9_-]+)*\s*", command):
        return None

    collection_event = None
    for event in query.get("prior_events", []):
        excerpt = event.get("result_excerpt")
        if isinstance(excerpt, str) and event.get("exit_code") not in (0, None):
            lowered = excerpt.lower()
            if "error collecting" in lowered or "errors during collection" in lowered:
                if collection_event is None or event.get("event_index", -1) > collection_event.get("event_index", -1):
                    collection_event = event

    if collection_event is None:
        return None

    install_event = None
    for event in query.get("prior_events", []):
        event_command = event.get("command")
        if not isinstance(event_command, str) or event.get("exit_code") != 0:
            continue
        if event.get("event_index", -1) <= collection_event.get("event_index", -1):
            continue
        if "pip install" in event_command or "apt-get install" in event_command:
            if install_event is None or event.get("event_index", -1) > install_event.get("event_index", -1):
                install_event = event

    if install_event is None:
        return None

    return {
        "rule_id": "full_suite_retry_after_collection_remediation",
        "state": "dependencies_remediated",
        "evidence_event_indices": [
            collection_event["event_index"],
            install_event["event_index"],
        ],
    }
