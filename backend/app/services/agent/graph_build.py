from langgraph.graph import END, START, StateGraph
from app.services.agent.graph_state import FitnessState

def build_graph(owner):
    builder = StateGraph(FitnessState)
    builder.add_node("preprocess", owner._preprocess_node)
    builder.add_node("load_context", owner._load_context_node)
    builder.add_node("dispatch", owner._dispatch_node)
    builder.add_node("safety_guard", owner._safety_guard_node)
    builder.add_node("workout_record", owner._run_workout_worker)
    builder.add_node("plan_adjust", owner._run_plan_worker)
    builder.add_node("nutrition", owner._run_nutrition_worker)
    builder.add_node("training_qa", owner._run_qa_worker)
    builder.add_node("post_worker", owner._post_worker_node)
    builder.add_node("finalize", owner._finalize_node)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "load_context")
    builder.add_edge("load_context", "dispatch")
    builder.add_conditional_edges("dispatch", owner._route_dispatch, {"safety_guard": "safety_guard", "workout_record": "workout_record", "plan_adjust": "plan_adjust", "nutrition": "nutrition", "training_qa": "training_qa", "finalize": "finalize"})
    for worker_name in ("workout_record", "plan_adjust", "nutrition", "training_qa"):
        builder.add_edge(worker_name, "post_worker")
    builder.add_conditional_edges("post_worker", owner._route_after_worker, {"dispatch": "dispatch", "finalize": "finalize"})
    builder.add_edge("safety_guard", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
