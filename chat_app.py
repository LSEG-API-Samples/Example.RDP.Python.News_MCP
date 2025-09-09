from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import uvicorn
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from llm import get_default_chat_llm

# Create FastAPI instance
app = FastAPI(title="LangGraph Chat Interface")


# Pydantic models for API
class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []


# Initialize LangGraph
async def get_news_tools():
    # Get absolute path to the news server script
    project_root = os.path.dirname(os.path.abspath(__file__))
    news_server_path = os.path.join(project_root, "mcp-servers/news-server.py")
    
    client = MultiServerMCPClient(
        {
            "news": {
                "command": "python",
                "args": [news_server_path],
                "transport": "stdio",
                "env": os.environ.copy(),
            }
        }
    )
    return await client.get_tools()


# Global variables for tools and graph
news_tools = None
all_tools = None
graph = None
llm_with_tools = None

# Define LLM
llm = get_default_chat_llm()

# Initialize async components
async def initialize_app():
    global news_tools, all_tools, graph, llm_with_tools

    # Load news tools from MCP server
    news_tools = await get_news_tools()
    all_tools = news_tools

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant with access to news tools. You can help users search for and analyze news content."
    )

    # Node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_app()
    yield
    # Shutdown (nothing to cleanup for now)


# Create FastAPI instance with lifespan
app = FastAPI(title="LangGraph Chat Interface", lifespan=lifespan)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Chat service is running"}


# Agent details endpoint
@app.get("/agent-info")
async def get_agent_info():
    if graph is None or all_tools is None:
        return {"status": "not_ready", "message": "Agent not initialized"}

    # Get LLM model info
    model_info = {
        "model_id": llm.model_id,
        "region": llm.region_name,
        "max_tokens": llm.model_kwargs.get("max_tokens", "Not set"),
        "temperature": llm.model_kwargs.get("temperature", "Not set"),
    }

    # Get available tools info
    tools_info = []
    for tool in all_tools:
        tool_info = {"name": tool.name, "description": tool.description}
        tools_info.append(tool_info)

    return {
        "status": "ready",
        "agent": {
            "name": "LSEG News Agent",
            "description": "AI assistant with access to news search and analysis tools",
        },
        "llm": model_info,
        "mcp_servers": {
            "news": {
                "description": "News search and retrieval server",
                "tools_count": len(tools_info),
            }
        },
        "tools": tools_info,
        "graph_nodes": ["assistant", "tools"],
    }


# Chat endpoint for REST API
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    if graph is None:
        raise HTTPException(
            status_code=503, detail="Service not ready - graph not initialized"
        )

    try:
        # Create initial state with user message
        initial_state = {"messages": [HumanMessage(content=chat_message.message)]}

        # Run the graph
        result = await graph.ainvoke(initial_state)

        # Extract the final response
        final_message = result["messages"][-1]
        response_content = final_message.content

        # Extract tool calls if any
        tool_calls = []
        for message in result["messages"]:
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "name": tool_call.get("name", ""),
                            "args": tool_call.get("args", {}),
                            "id": tool_call.get("id", ""),
                        }
                    )

        return ChatResponse(response=response_content, tool_calls=tool_calls)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# WebSocket endpoint for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if graph is None:
        await websocket.send_text(
            json.dumps(
                {"error": "Service not ready - graph not initialized", "type": "error"}
            )
        )
        await websocket.close()
        return

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            if not user_message:
                await websocket.send_text(
                    json.dumps({"error": "Empty message received"})
                )
                continue

            try:
                # Create initial state with user message
                initial_state = {"messages": [HumanMessage(content=user_message)]}

                # Send initial thinking message
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "thinking",
                            "content": "🤔 AI is analyzing your request...",
                        }
                    )
                )

                # Stream the graph execution
                reasoning_steps = []
                tool_calls = []
                final_response = ""
                step_counter = 0

                async for chunk in graph.astream(initial_state):
                    for node_name, node_output in chunk.items():
                        step_counter += 1

                        if node_name == "assistant":
                            # AI is thinking/planning
                            messages = node_output.get("messages", [])
                            if messages:
                                last_message = messages[-1]

                                # Check if AI is making tool calls
                                if (
                                    hasattr(last_message, "tool_calls")
                                    and last_message.tool_calls
                                ):
                                    reasoning_step = {
                                        "step": step_counter,
                                        "type": "reasoning",
                                        "content": f"AI is planning to use {len(last_message.tool_calls)} tool(s)",
                                    }
                                    reasoning_steps.append(reasoning_step)

                                    # Send real-time update
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "reasoning_step",
                                                "step": reasoning_step,
                                            }
                                        )
                                    )

                                    # Process each tool call
                                    for tool_call in last_message.tool_calls:
                                        step_counter += 1
                                        tool_info = {
                                            "name": tool_call.get("name", ""),
                                            "args": tool_call.get("args", {}),
                                            "id": tool_call.get("id", ""),
                                        }
                                        tool_calls.append(tool_info)

                                        tool_step = {
                                            "step": step_counter,
                                            "type": "tool_call",
                                            "tool_name": tool_info["name"],
                                            "content": f"Calling tool: {tool_info['name']}",
                                            "args": tool_info["args"],
                                        }
                                        reasoning_steps.append(tool_step)

                                        # Send real-time update
                                        await websocket.send_text(
                                            json.dumps(
                                                {
                                                    "type": "reasoning_step",
                                                    "step": tool_step,
                                                }
                                            )
                                        )

                                # Check if this is the final response (AI message with content but no actual tool calls)
                                elif (
                                    hasattr(last_message, "content")
                                    and last_message.content
                                ):
                                    # This is the final response
                                    if (
                                        not final_response
                                    ):  # Only set if we haven't captured it yet
                                        final_response = last_message.content

                                    final_step = {
                                        "step": step_counter,
                                        "type": "final_response",
                                        "content": "AI has generated final response",
                                    }
                                    reasoning_steps.append(final_step)

                                    # Send real-time update
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "reasoning_step",
                                                "step": final_step,
                                            }
                                        )
                                    )

                        elif node_name == "tools":
                            # Tools are executing
                            messages = node_output.get("messages", [])
                            if messages:
                                for message in messages:
                                    if hasattr(message, "content"):
                                        step_counter += 1
                                        tool_response_step = {
                                            "step": step_counter,
                                            "type": "tool_response",
                                            "content": "Tool response received",
                                            "response": (
                                                message.content[:200] + "..."
                                                if len(str(message.content)) > 200
                                                else str(message.content)
                                            ),
                                        }
                                        reasoning_steps.append(tool_response_step)

                                        # Send real-time update
                                        await websocket.send_text(
                                            json.dumps(
                                                {
                                                    "type": "reasoning_step",
                                                    "step": tool_response_step,
                                                }
                                            )
                                        )

                # Send final complete response
                await websocket.send_text(
                    json.dumps(
                        {
                            "response": final_response
                            or "I apologize, but I wasn't able to generate a response.",
                            "reasoning_steps": reasoning_steps,
                            "tool_calls": tool_calls,
                            "type": "final_complete",
                        }
                    )
                )

            except Exception as e:
                await websocket.send_text(
                    json.dumps({"error": f"Chat error: {str(e)}", "type": "error"})
                )

    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Chat</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                height: calc(100vh - 40px);
                display: flex;
                flex-direction: column;
            }
            .chat-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                flex: 1;
                display: flex;
                flex-direction: column;
                min-height: 0;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #fafafa;
                border-radius: 5px;
                min-height: 0;
            }
            .message {
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 8px;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                text-align: right;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #e9ecef;
                color: #333;
                margin-right: 20%;
            }
            .reasoning-container {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin: 10px 0;
                overflow: hidden;
            }
            .reasoning-header {
                background-color: #6c757d;
                color: white;
                padding: 8px 12px;
                font-weight: bold;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .reasoning-content {
                padding: 10px;
                display: none;
            }
            .reasoning-content.expanded {
                display: block;
            }
            .reasoning-step {
                margin: 8px 0;
                padding: 8px;
                border-radius: 4px;
                border-left: 3px solid #007bff;
                background-color: white;
            }
            .reasoning-step.tool_call {
                border-left-color: #28a745;
                background-color: #f8fff9;
            }
            .reasoning-step.tool_response {
                border-left-color: #ffc107;
                background-color: #fffbf0;
            }
            .reasoning-step.final_response {
                border-left-color: #dc3545;
                background-color: #fff5f5;
            }
            .step-header {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .step-content {
                font-size: 0.9em;
                color: #555;
            }
            .tool-args {
                background-color: #f1f3f4;
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
                font-family: monospace;
                font-size: 0.8em;
                max-height: 100px;
                overflow-y: auto;
            }
            .tool-response {
                background-color: #f1f3f4;
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
                font-family: monospace;
                font-size: 0.8em;
                max-height: 100px;
                overflow-y: auto;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            #messageInput {
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            #sendButton {
                padding: 12px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #sendButton:hover {
                background-color: #0056b3;
            }
            #sendButton:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .status {
                text-align: center;
                color: #6c757d;
                font-style: italic;
                margin-bottom: 10px;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .toggle-icon {
                transition: transform 0.2s;
            }
            .toggle-icon.expanded {
                transform: rotate(90deg);
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>LSEG News Chat</h1>
            <div id="status" class="status">Connected</div>
            <div id="chatMessages" class="chat-messages">
                <div class="message bot-message">
                    Hello! I'm your news assistant. I can help you search for and analyze news content. What would you like to know?
                </div>
            </div>
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type your message here..." 
                       onkeypress="handleKeyPress(event)">
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
            const messagesDiv = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const statusDiv = document.getElementById('status');

            ws.onopen = function(event) {
                // Fetch agent information and display detailed status
                fetch('/agent-info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'ready') {
                            const agentInfo = `Connected to ${data.agent.name} | Model: ${data.llm.model_id} | Tools: ${data.tools.length} available`;
                            statusDiv.innerHTML = agentInfo;
                            statusDiv.style.color = '#28a745';
                            statusDiv.style.fontSize = '12px';
                            
                            // Add detailed info as a welcome message
                            addAgentInfoMessage(data);
                        } else {
                            statusDiv.textContent = 'Connected to Agent (Initializing...)';
                            statusDiv.style.color = '#ffc107';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching agent info:', error);
                        statusDiv.textContent = 'Connected to Agent';
                        statusDiv.style.color = '#28a745';
                    });
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                console.log("Received message:", data);
                
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot-message');
                    resetStreamingState();
                } else if (data.type === 'thinking') {
                    // Show initial thinking message
                    const thinkingDiv = document.createElement('div');
                    thinkingDiv.className = 'message bot-message thinking-message';
                    thinkingDiv.textContent = data.content;
                    thinkingDiv.id = 'current-thinking';
                    messagesDiv.appendChild(thinkingDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                } else if (data.type === 'reasoning_step') {
                    // Add or update reasoning steps in real-time
                    handleReasoningStep(data.step);
                } else if (data.type === 'final_complete') {
                    // Complete the message with final response
                    completeFinalMessage(data);
                } else if (data.type === 'detailed_message') {
                    // Fallback for non-streaming mode
                    addDetailedMessage(data);
                } else if (data.response) {
                    addMessage(data.response, 'bot-message');
                }
                
                if (data.type === 'final_complete' || data.type === 'detailed_message' || data.error) {
                    sendButton.disabled = false;
                    sendButton.textContent = 'Send';
                }
            };

            ws.onclose = function(event) {
                statusDiv.textContent = 'Agent Disconnected';
                statusDiv.style.color = '#dc3545';
                statusDiv.style.fontSize = '14px';
            };

            ws.onerror = function(error) {
                statusDiv.textContent = 'Agent Connection Error';
                statusDiv.style.color = '#dc3545';
                statusDiv.style.fontSize = '14px';
            };

            let currentReasoningContainer = null;
            let currentReasoningContent = null;
            let currentMessageDiv = null;

            function addAgentInfoMessage(agentData) {
                const infoDiv = document.createElement('div');
                infoDiv.className = 'message bot-message';
                infoDiv.style.backgroundColor = '#e3f2fd';
                infoDiv.style.border = '1px solid #2196f3';
                
                let infoHtml = `
                    <div style="font-weight: bold; margin-bottom: 10px;">🤖 Agent Information</div>
                    <div><strong>Agent:</strong> ${agentData.agent.name}</div>
                    <div><strong>Description:</strong> ${agentData.agent.description}</div>
                    <div style="margin-top: 10px;"><strong>LLM Model:</strong></div>
                    <div style="margin-left: 15px;">
                        <div>• Model: ${agentData.llm.model_id}</div>
                        <div>• Region: ${agentData.llm.region}</div>
                        <div>• Max Tokens: ${agentData.llm.max_tokens}</div>
                        <div>• Temperature: ${agentData.llm.temperature}</div>
                    </div>
                    <div style="margin-top: 10px;"><strong>MCP Servers:</strong></div>
                    <div style="margin-left: 15px;">
                        <div>• News Server: ${agentData.mcp_servers.news.description} (${agentData.mcp_servers.news.tools_count} tools)</div>
                    </div>
                    <div style="margin-top: 10px;"><strong>Available Tools:</strong></div>
                    <div style="margin-left: 15px;">
                `;
                
                agentData.tools.forEach(tool => {
                    infoHtml += `<div>• ${tool.name}: ${tool.description}</div>`;
                });
                
                infoHtml += `</div>`;
                
                infoDiv.innerHTML = infoHtml;
                messagesDiv.appendChild(infoDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            // Simple markdown to HTML converter
            function convertMarkdownToHtml(text) {
                let html = text;
                
                try {
                    // Convert headers (### ## #) - must be at start of line
                    html = html.replace(/^### (.+)$/gm, '<h3 style="color: #2c3e50; margin: 15px 0 10px 0; font-size: 1.2em; font-weight: bold;">$1</h3>');
                    html = html.replace(/^## (.+)$/gm, '<h2 style="color: #2c3e50; margin: 15px 0 10px 0; font-size: 1.3em; font-weight: bold;">$1</h2>');
                    html = html.replace(/^# (.+)$/gm, '<h1 style="color: #2c3e50; margin: 15px 0 10px 0; font-size: 1.4em; font-weight: bold;">$1</h1>');
                    
                    // Convert bold text (**text**) - safer pattern
                    html = html.replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong style="font-weight: bold; color: #2c3e50;">$1</strong>');
                    
                    // Convert italic text (*text*) - safer pattern  
                    html = html.replace(/\\*([^\\*]+)\\*/g, '<em style="font-style: italic;">$1</em>');
                    
                    // Convert numbered lists (1. 2. 3.) - at start of line
                    html = html.replace(/^(\\d+)\\. (.+)$/gm, '<div style="margin: 8px 0; padding-left: 20px;"><span style="font-weight: bold; color: #007bff; margin-right: 8px;">$1.</span>$2</div>');
                    
                    // Convert bullet points (- or *) - at start of line
                    html = html.replace(/^- (.+)$/gm, '<div style="margin: 8px 0; padding-left: 20px;"><span style="color: #007bff; margin-right: 8px;">•</span>$1</div>');
                    
                    // Convert double line breaks to paragraph breaks
                    html = html.replace(/\\n\\n/g, '</p><p style="margin: 10px 0; line-height: 1.5;">');
                    
                    // Convert single line breaks to <br>
                    html = html.replace(/\\n/g, '<br>');
                    
                    // Wrap in paragraph tags if not empty
                    if (html.trim()) {
                        html = '<div style="margin: 10px 0; line-height: 1.5;">' + html + '</div>';
                    }
                    
                } catch (error) {
                    console.error('Markdown conversion error:', error);
                    // Fallback: just replace line breaks and bold text
                    html = text
                        .replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong>$1</strong>')
                        .replace(/\\n/g, '<br>');
                }
                
                return html;
            }

            function resetStreamingState() {
                currentReasoningContainer = null;
                currentReasoningContent = null;
                currentMessageDiv = null;
                
                // Remove any leftover thinking messages
                const thinkingMsg = document.getElementById('current-thinking');
                if (thinkingMsg) {
                    thinkingMsg.remove();
                }
            }

            function handleReasoningStep(step) {
                // Initialize reasoning container if needed
                if (!currentReasoningContainer) {
                    // Remove thinking message
                    const thinkingMsg = document.getElementById('current-thinking');
                    if (thinkingMsg) {
                        thinkingMsg.remove();
                    }
                    
                    // Create main message div
                    currentMessageDiv = document.createElement('div');
                    currentMessageDiv.className = 'message bot-message';
                    
                    // Create reasoning container
                    currentReasoningContainer = document.createElement('div');
                    currentReasoningContainer.className = 'reasoning-container';
                    
                    const reasoningHeader = document.createElement('div');
                    reasoningHeader.className = 'reasoning-header';
                    reasoningHeader.onclick = () => toggleReasoning(reasoningHeader);
                    reasoningHeader.innerHTML = `
                        <span>🧠 Model Reasoning & Tool Calls (0 steps)</span>
                        <span class="toggle-icon expanded">▼</span>
                    `;
                    
                    currentReasoningContent = document.createElement('div');
                    currentReasoningContent.className = 'reasoning-content expanded';
                    
                    currentReasoningContainer.appendChild(reasoningHeader);
                    currentReasoningContainer.appendChild(currentReasoningContent);
                    currentMessageDiv.appendChild(currentReasoningContainer);
                    messagesDiv.appendChild(currentMessageDiv);
                }
                
                // Add the new step
                const stepDiv = document.createElement('div');
                stepDiv.className = `reasoning-step ${step.type}`;
                
                const stepHeader = document.createElement('div');
                stepHeader.className = 'step-header';
                
                let icon = '';
                switch(step.type) {
                    case 'reasoning': icon = '🤔'; break;
                    case 'tool_call': icon = '🔧'; break;
                    case 'tool_response': icon = '📋'; break;
                    case 'final_response': icon = '💬'; break;
                }
                
                stepHeader.textContent = `${icon} Step ${step.step}: ${step.content}`;
                stepDiv.appendChild(stepHeader);
                
                if (step.tool_name) {
                    const toolName = document.createElement('div');
                    toolName.className = 'step-content';
                    toolName.innerHTML = `<strong>Tool:</strong> ${step.tool_name}`;
                    stepDiv.appendChild(toolName);
                }
                
                if (step.args) {
                    const argsDiv = document.createElement('div');
                    argsDiv.className = 'tool-args';
                    argsDiv.textContent = JSON.stringify(step.args, null, 2);
                    stepDiv.appendChild(argsDiv);
                }
                
                if (step.response) {
                    const responseDiv = document.createElement('div');
                    responseDiv.className = 'tool-response';
                    responseDiv.textContent = step.response;
                    stepDiv.appendChild(responseDiv);
                }
                
                currentReasoningContent.appendChild(stepDiv);
                
                // Update step count in header - find the title within the current container
                const title = currentReasoningContainer.querySelector('.reasoning-header span:first-child');
                if (title) {
                    const stepCount = currentReasoningContent.children.length;
                    title.textContent = `🧠 Model Reasoning & Tool Calls (${stepCount} steps)`;
                }
                
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function completeFinalMessage(data) {
                console.log("completeFinalMessage called with:", data);
                
                // Remove any leftover thinking message
                const thinkingMsg = document.getElementById('current-thinking');
                if (thinkingMsg) {
                    thinkingMsg.remove();
                }
                
                if (!currentMessageDiv) {
                    // If no reasoning steps, create a simple message
                    currentMessageDiv = document.createElement('div');
                    currentMessageDiv.className = 'message bot-message';
                    messagesDiv.appendChild(currentMessageDiv);
                }
                
                if (data.response && data.response.trim()) {
                    // Add the final response below the reasoning container
                    const responseDiv = document.createElement('div');
                    responseDiv.className = 'final-response-text';
                    responseDiv.style.marginTop = '10px';
                    responseDiv.style.fontWeight = 'normal';
                    responseDiv.style.padding = '15px';
                    responseDiv.style.backgroundColor = '#f8f9fa';
                    responseDiv.style.borderRadius = '8px';
                    responseDiv.style.border = '1px solid #dee2e6';
                    responseDiv.style.lineHeight = '1.6';
                    
                    // Convert markdown to HTML
                    responseDiv.innerHTML = convertMarkdownToHtml(data.response);
                    
                    // Add at the end of the message (below reasoning)
                    currentMessageDiv.appendChild(responseDiv);
                } else {
                    // Fallback if no response
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error-message';
                    errorDiv.style.color = 'red';
                    errorDiv.style.fontStyle = 'italic';
                    errorDiv.textContent = 'No response received from AI';
                    currentMessageDiv.appendChild(errorDiv);
                }
                
                // Reset for next message
                resetStreamingState();
                
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function addMessage(content, className) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + className;
                
                // Use markdown conversion for bot messages, plain text for user messages
                if (className.includes('bot-message')) {
                    messageDiv.innerHTML = convertMarkdownToHtml(content);
                } else {
                    messageDiv.textContent = content;
                }
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function addDetailedMessage(data) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot-message';
                
                // Add the main response
                const responseDiv = document.createElement('div');
                responseDiv.innerHTML = convertMarkdownToHtml(data.response);
                messageDiv.appendChild(responseDiv);
                
                // Add reasoning steps if available
                if (data.reasoning_steps && data.reasoning_steps.length > 0) {
                    const reasoningContainer = document.createElement('div');
                    reasoningContainer.className = 'reasoning-container';
                    
                    const reasoningHeader = document.createElement('div');
                    reasoningHeader.className = 'reasoning-header';
                    reasoningHeader.onclick = () => toggleReasoning(reasoningHeader);
                    reasoningHeader.innerHTML = `
                        <span>🧠 Model Reasoning & Tool Calls (${data.reasoning_steps.length} steps)</span>
                        <span class="toggle-icon">▶</span>
                    `;
                    
                    const reasoningContent = document.createElement('div');
                    reasoningContent.className = 'reasoning-content';
                    
                    data.reasoning_steps.forEach(step => {
                        const stepDiv = document.createElement('div');
                        stepDiv.className = `reasoning-step ${step.type}`;
                        
                        const stepHeader = document.createElement('div');
                        stepHeader.className = 'step-header';
                        
                        let icon = '';
                        switch(step.type) {
                            case 'reasoning': icon = '🤔'; break;
                            case 'tool_call': icon = '🔧'; break;
                            case 'tool_response': icon = '📋'; break;
                            case 'final_response': icon = '💬'; break;
                        }
                        
                        stepHeader.textContent = `${icon} Step ${step.step}: ${step.content}`;
                        stepDiv.appendChild(stepHeader);
                        
                        if (step.tool_name) {
                            const toolName = document.createElement('div');
                            toolName.className = 'step-content';
                            toolName.innerHTML = `<strong>Tool:</strong> ${step.tool_name}`;
                            stepDiv.appendChild(toolName);
                        }
                        
                        if (step.args) {
                            const argsDiv = document.createElement('div');
                            argsDiv.className = 'tool-args';
                            argsDiv.textContent = JSON.stringify(step.args, null, 2);
                            stepDiv.appendChild(argsDiv);
                        }
                        
                        if (step.response) {
                            const responseDiv = document.createElement('div');
                            responseDiv.className = 'tool-response';
                            responseDiv.textContent = step.response;
                            stepDiv.appendChild(responseDiv);
                        }
                        
                        reasoningContent.appendChild(stepDiv);
                    });
                    
                    reasoningContainer.appendChild(reasoningHeader);
                    reasoningContainer.appendChild(reasoningContent);
                    messageDiv.appendChild(reasoningContainer);
                }
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function toggleReasoning(header) {
                const content = header.nextElementSibling;
                const icon = header.querySelector('.toggle-icon');
                
                if (content.classList.contains('expanded')) {
                    content.classList.remove('expanded');
                    icon.classList.remove('expanded');
                } else {
                    content.classList.add('expanded');
                    icon.classList.add('expanded');
                }
            }

            function sendMessage() {
                const message = messageInput.value.trim();
                if (message && ws.readyState === WebSocket.OPEN) {
                    addMessage(message, 'user-message');
                    ws.send(JSON.stringify({message: message}));
                    messageInput.value = '';
                    sendButton.disabled = true;
                    sendButton.textContent = 'Sending...';
                }
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Focus on input when page loads
            messageInput.focus();
        </script>
    </body>
    </html>
    """


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
