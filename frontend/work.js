// Base API URL - Change this to your production backend URL if deployed
const API_BASE_URL = "http://127.0.0.1:8000";

// DOM Elements
const fileUploadInput = document.getElementById("file-upload");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const chatHistory = document.getElementById("chat-history");
const statusBadge = document.getElementById("status");

// Session State
let primaryDatasetId = null;
let predictionDatasetId = null;

// Initialize event listeners
document.addEventListener("DOMContentLoaded", () => { //waiting until the webpage is completely loaded
    fileUploadInput.addEventListener("change", handleFileUpload);
    sendBtn.addEventListener("click", handleSendQuery);
    queryInput.addEventListener("keydown", (e) => { //if user pressed enter button
        if (e.key === "Enter" && !queryInput.disabled) {
            handleSendQuery();
        }
    });
});


//Handles the CSV file uploads. Supports up to 2 files.
 
async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    // Reset previous states
    primaryDatasetId = null;
    predictionDatasetId = null;
    updateStatusBadge("Uploading...", "offline");

    addSystemMessage("Uploading and processing your dataset(s)... Please wait.");

    // Limit files to maximum of 2
    const filesToUpload = Array.from(files).slice(0, 2);

    try {
        // Upload the primary dataset
        const primaryRes = await uploadFile(filesToUpload[0]);
        primaryDatasetId = primaryRes.dataset_id;
        addSystemMessage(`✅ Primary dataset loaded: <b>${primaryRes.filename}</b> (${primaryRes.row_count} rows).`);

        // If there's a second file, upload it as the prediction dataset
        if (filesToUpload.length > 1) {
            const predRes = await uploadFile(filesToUpload[1]);
            predictionDatasetId = predRes.dataset_id;
            addSystemMessage(`✅ Prediction dataset loaded: <b>${predRes.filename}</b> (${predRes.row_count} rows).`);
            updateStatusBadge("2 Datasets Active", "online");
        } else {
            updateStatusBadge("Dataset Active", "online");
        }

        // Enable inputs for querying
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.placeholder = "Ask a question about your dataset...";
        queryInput.focus();

    } catch (error) {
        console.error("Upload Error:", error);
        addSystemMessage(`❌ Upload failed: ${error.message}`);
        updateStatusBadge("Upload Failed", "offline");
        disableInputs();
    }
}

/**
 * Performs the actual multipart/form-data POST request to FastAPI /upload
 */
async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error occurred during upload.");
    }

    return await response.json();
}

/**
 * Handles sending a user query and rendering the response
 */
async function handleSendQuery() {
    const query = queryInput.value.trim();
    if (!query || !primaryDatasetId) return;

    // Append user message to chat
    addUserMessage(query);
    queryInput.value = ""; // Clear input

    // Create a temporary "AI is thinking..." typing indicator
    const thinkingId = addAIThinkingIndicator();

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                dataset_id: primaryDatasetId,
                query: query,
                prediction_dataset_id: predictionDatasetId
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Error processing your request.");
        }

        const resData = await response.json();
        
        // Remove the loading indicator
        removeElement(thinkingId);

        // Display results
        renderAIResponse(resData.data);

    } catch (error) {
        console.error("Analysis Error:", error);
        removeElement(thinkingId);
        addSystemMessage(`❌ Analysis failed: ${error.message}`);
    }
}

/* ==========================================
   UI HELPER FUNCTIONS
   ========================================== */

function updateStatusBadge(text, statusClass) {
    statusBadge.textContent = text;
    statusBadge.className = `badge ${statusClass}`;
}

function disableInputs() {
    queryInput.disabled = true;
    sendBtn.disabled = true;
    queryInput.placeholder = "Upload a dataset to ask questions...";
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight; //when a new message come automatically scrolls 
}

function addUserMessage(message) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message-container user";
    msgDiv.innerHTML = `<div class="message user-message">${escapeHtml(message)}</div>`;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
}

function addSystemMessage(htmlContent) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message-container system";
    msgDiv.innerHTML = `<div class="message system-message">${htmlContent}</div>`;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
}

function addAIThinkingIndicator() {
    const id = "thinking-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = "message-container ai";
    msgDiv.id = id;
    msgDiv.innerHTML = `<div class="message ai-message loading">AI is analyzing your data... ⏳</div>`;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * Parses and displays AI outputs (text, markdown-styled text, and Plotly charts)
 */
function renderAIResponse(data) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message-container ai";
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message ai-message";

    // 1. Render text answers (if your backend returns strings)
    if (typeof data === "string") {
        contentDiv.innerHTML = data.replace(/\n/g, "<br>");
    } else if (data && typeof data === "object") {
        // If your pipeline returns structured JSON, customize this mapping.
        // Assuming data structure: { explanation: "...", plot: { data: [...], layout: {} } }
        let htmlContent = "";
        if (data.explanation) {
            htmlContent += `<p>${data.explanation.replace(/\n/g, "<br>")}</p>`;
        } else if (data.summary) {
            htmlContent += `<p>${data.summary.replace(/\n/g, "<br>")}</p>`;
        } else {
            htmlContent += `<p>Analysis completed successfully.</p>`;
        }
        contentDiv.innerHTML = htmlContent;

        // 2. Render Plotly charts dynamically if "plot" or "chart" JSON is present
        const chartData = data.plot || data.chart;
        if (chartData && chartData.data && chartData.layout) {
            const chartDiv = document.createElement("div");
            const chartId = "plotly-chart-" + Date.now(); //creating new unique id 
            chartDiv.id = chartId;
            chartDiv.style.width = "100%";
            chartDiv.style.marginTop = "15px";
            contentDiv.appendChild(chartDiv);

            // Wait for html to create chart container i.e div then plotly can safely draw to avoid crashes
            setTimeout(() => {
                // Ensure Plotly does not break the layout of dark-themed pages
                const layout = {
                    ...chartData.layout,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#e2e8f0', ...chartData.layout?.font }
                };
                Plotly.newPlot(chartId, chartData.data, layout, { responsive: true });
            }, 50);
        }
    }

    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
}
/*
for safety ,if user wrote html or js scripts it will be shown as 
plain text instead of being executed by the browser 
*/
function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}