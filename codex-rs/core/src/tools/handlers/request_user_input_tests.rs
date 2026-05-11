use super::*;
use crate::session::tests::make_session_and_context;
use crate::session::tests::make_session_and_context_with_rx;
use crate::session::turn_context::TurnContext;
use crate::state::ActiveTurn;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::turn_diff_tracker::TurnDiffTracker;
use crate::turn_metadata::McpTurnMetadataContext;
use crate::turn_metadata::PRIOR_USER_INPUT_REQUESTED_KEY;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ModeKind;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::request_user_input::RequestUserInputAnswer;
use codex_protocol::request_user_input::RequestUserInputArgs;
use codex_protocol::request_user_input::RequestUserInputQuestion;
use codex_protocol::request_user_input::RequestUserInputQuestionOption;
use codex_protocol::request_user_input::RequestUserInputResponse;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

fn request_user_input_args() -> RequestUserInputArgs {
    RequestUserInputArgs {
        questions: vec![RequestUserInputQuestion {
            id: "pick_one".to_string(),
            header: "Hdr".to_string(),
            question: "Pick one".to_string(),
            is_other: false,
            is_secret: false,
            options: Some(vec![
                RequestUserInputQuestionOption {
                    label: "A".to_string(),
                    description: "A".to_string(),
                },
                RequestUserInputQuestionOption {
                    label: "B".to_string(),
                    description: "B".to_string(),
                },
            ]),
        }],
    }
}

fn request_user_input_response() -> RequestUserInputResponse {
    RequestUserInputResponse {
        answers: HashMap::from([(
            "pick_one".to_string(),
            RequestUserInputAnswer {
                answers: vec!["A".to_string()],
            },
        )]),
    }
}

fn prior_user_input_requested(turn: &TurnContext) -> Option<bool> {
    let meta = turn
        .turn_metadata_state
        .current_meta_value_for_mcp_request(McpTurnMetadataContext {
            model: turn.model_info.slug.as_str(),
            reasoning_effort: turn.effective_reasoning_effort(),
        })
        .expect("turn metadata should be present");
    meta.get(PRIOR_USER_INPUT_REQUESTED_KEY)
        .and_then(serde_json::Value::as_bool)
}

fn request_user_input_tool_invocation(
    session: Arc<crate::session::session::Session>,
    turn: Arc<TurnContext>,
) -> ToolInvocation {
    ToolInvocation {
        session,
        turn,
        cancellation_token: tokio_util::sync::CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::default())),
        call_id: "call-1".to_string(),
        tool_name: codex_tools::ToolName::plain(REQUEST_USER_INPUT_TOOL_NAME),
        source: crate::tools::context::ToolCallSource::Direct,
        payload: ToolPayload::Function {
            arguments: serde_json::to_string(&request_user_input_args())
                .expect("serialize request_user_input args"),
        },
    }
}

#[tokio::test]
async fn multi_agent_v2_request_user_input_rejects_subagent_threads() {
    let (session, mut turn) = make_session_and_context().await;
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: ThreadId::new(),
        depth: 1,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    });

    let result = RequestUserInputHandler {
        available_modes: Vec::new(),
    }
    .handle(request_user_input_tool_invocation(
        Arc::new(session),
        Arc::new(turn),
    ))
    .await;

    let Err(err) = result else {
        panic!("sub-agent request_user_input should fail");
    };
    assert_eq!(
        err,
        FunctionCallError::RespondToModel(
            "request_user_input can only be used by the root thread".to_string(),
        )
    );
}

#[tokio::test]
async fn request_user_input_handler_marks_prior_user_input_for_mcp_metadata() {
    let (session, turn, rx_event) = make_session_and_context_with_rx().await;
    *session.active_turn.lock().await = Some(ActiveTurn::default());

    assert_eq!(prior_user_input_requested(&turn), None);

    let handler_task = tokio::spawn({
        let session = Arc::clone(&session);
        let turn = Arc::clone(&turn);
        async move {
            RequestUserInputHandler {
                available_modes: vec![ModeKind::Default],
            }
            .handle(request_user_input_tool_invocation(session, turn))
            .await
        }
    });

    let event = tokio::time::timeout(Duration::from_secs(1), rx_event.recv())
        .await
        .expect("request_user_input event timed out")
        .expect("expected request_user_input event");
    let EventMsg::RequestUserInput(request) = event.msg else {
        panic!("expected request_user_input event");
    };

    assert_eq!(prior_user_input_requested(&turn), Some(true));

    session
        .notify_user_input_response(&request.turn_id, request_user_input_response())
        .await;

    tokio::time::timeout(Duration::from_secs(1), handler_task)
        .await
        .expect("request_user_input handler timed out")
        .expect("request_user_input handler task failed")
        .expect("request_user_input handler should succeed");
}

#[tokio::test]
async fn lower_level_session_request_user_input_does_not_mark_prior_user_input() {
    let (session, turn, rx_event) = make_session_and_context_with_rx().await;
    *session.active_turn.lock().await = Some(ActiveTurn::default());

    let request_task = tokio::spawn({
        let session = Arc::clone(&session);
        let turn = Arc::clone(&turn);
        async move {
            session
                .request_user_input(
                    turn.as_ref(),
                    "call-1".to_string(),
                    request_user_input_args(),
                )
                .await
        }
    });

    let event = tokio::time::timeout(Duration::from_secs(1), rx_event.recv())
        .await
        .expect("request_user_input event timed out")
        .expect("expected request_user_input event");
    let EventMsg::RequestUserInput(request) = event.msg else {
        panic!("expected request_user_input event");
    };

    assert_eq!(prior_user_input_requested(&turn), None);

    session
        .notify_user_input_response(&request.turn_id, request_user_input_response())
        .await;

    tokio::time::timeout(Duration::from_secs(1), request_task)
        .await
        .expect("request_user_input timed out")
        .expect("request_user_input task failed")
        .expect("request_user_input should receive response");

    assert_eq!(prior_user_input_requested(&turn), None);
}
