// Base API URL
const API_BASE_URL = "http://127.0.0.1:8000";

// DOM Elements
const fileUploadInput = document.getElementById("file-upload");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const chatHistory = document.getElementById("chat-history");
const statusBadge = document.getElementById("status");

// Session State - Protected from accidental clearing
let primaryDatasetId = null;
let predictionDatasetId = null;

// Initialize event listeners
document.addEventListener("DOMContentLoaded", () => {
    // We listen to changes on the file input, but handle it with strict state validation
    fileUploadInput.addEventListener("change", handleFileUpload);
    
    sendBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        handleSendQuery();
    });

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !queryInput.disabled) {
            e.preventDefault();
            e.stopPropagation();
            handleSendQuery();
        }
    });
});

/**
 * Handles the CSV file uploads safely.
 */
async function handleFileUpload(event) {
    event.preventDefault();
    event.stopPropagation();

    const files = event.target.files;
    
    // SAFETY: If the event fired but no files were actually selected, 
    // immediately exit WITHOUT wiping out the existing session state!
    if (!files || files.length === 0) {
        return; 
    }

    // A valid upload has started, now we can safely prepare to update the state
    const filesToUpload = Array.from(files).slice(0, 2);
    
    // Visually show progress without breaking old active states until we succeed
    updateStatusBadge("Uploading...", "offline");
    addSystemMessage("Uploading and processing your dataset(s)... Please wait.");

    try {
        // 1. Upload the primary dataset
        const primaryRes = await uploadFile(filesToUpload[0]);
        primaryDatasetId = primaryRes.dataset_id; // Set state only after API success
        addSystemMessage(`✅ Primary dataset loaded: <b>${primaryRes.filename}</b> (${primaryRes.row_count} rows).`);

        // 2. If there's a second file, upload it as the prediction dataset
        if (filesToUpload.length > 1) {
            const predRes = await uploadFile(filesToUpload[1]);
            predictionDatasetId = predRes.dataset_id;
            addSystemMessage(`✅ Prediction dataset loaded: <b>${predRes.filename}</b> (${predRes.row_count} rows).`);
            updateStatusBadge("2 Datasets Active", "online");
        } else {
            predictionDatasetId = null; // Clear secondary if only one was uploaded this turn
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
        
        // Only disable inputs if we don't have a previously working dataset loaded
        if (!primaryDatasetId) {
            updateStatusBadge("Upload Failed", "offline");
            disableInputs();
        } else {
            // Keep the previous dataset active if the new upload attempt simply errored out
            updateStatusBadge(predictionDatasetId ? "2 Datasets Active" : "Dataset Active", "online");
        }
    } finally {
        // Clear the file input's value. 
        // This allows the user to re-upload the same file later if they want to,
        // without triggering false "change" events that reset the page.
        fileUploadInput.value = "";
    }
}

/**
 * Performs the multipart/form-data POST request to FastAPI /upload
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
    chatHistory.scrollTop = chatHistory.scrollHeight;
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
 * Parses and displays AI outputs (text and Plotly charts)
 */
function renderAIResponse(data) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message-container ai";
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message ai-message";

    // --- BULLETPROOF TYPE HANDLING ---
    if (data === null || data === undefined) {
        contentDiv.innerHTML = "<p>No data returned from analysis.</p>";
    } 
    // 1. Handle Numbers (e.g., 14.7)
    else if (typeof data === "number") {
        contentDiv.innerHTML = `<p><b>Result:</b> ${data}</p>`;
    } 
    // 2. Handle Strings
    else if (typeof data === "string") {
        contentDiv.innerHTML = data.replace(/\n/g, "<br>");
    } 
    // 3. Handle Arrays (e.g., [14.7, 34.7])
    else if (Array.isArray(data)) {
        const listItems = data.map(item => `<li>${item}</li>`).join("");
        contentDiv.innerHTML = `<ul style="margin: 0; padding-left: 20px;">${listItems}</ul>`;
    } 
    // 4. Handle Objects (Charts, Tables, Summaries)
    else if (typeof data === "object") {
        let htmlContent = "";
        
        // Render textual summary or explanation
        if (data.explanation) {
            htmlContent += `<p>${data.explanation.replace(/\n/g, "<br>")}</p>`;
        } else if (data.summary) {
            htmlContent += `<p>${data.summary.replace(/\n/g, "<br>")}</p>`;
        } else {
            htmlContent += `<p>Analysis completed successfully.</p>`;
        }
        contentDiv.innerHTML = htmlContent;

        // --- DYNAMIC TABLE RENDERING ---
        if (data.tables && data.tables.length > 0) {
            data.tables.forEach(tableObj => {
                const tableRows = tableObj.data;
                if (!tableRows || tableRows.length === 0) return;

                const headers = Object.keys(tableRows[0]);
                const tableContainer = document.createElement("div");
                tableContainer.className = "table-container";
                tableContainer.style.overflowX = "auto";
                tableContainer.style.marginTop = "15px";
                tableContainer.style.borderRadius = "8px";

                let tableHtml = `<table class="analysis-table" style="width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 14px; color: #e2e8f0; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);">`;
                
                tableHtml += `<thead><tr style="background: rgba(255, 255, 255, 0.1); text-align: left; border-bottom: 2px solid rgba(255, 255, 255, 0.15);">`;
                headers.forEach(header => {
                    const displayHeader = header.replace(/_/g, " ").toUpperCase();
                    tableHtml += `<th style="padding: 10px 12px; font-weight: 600;">${displayHeader}</th>`;
                });
                tableHtml += `</tr></thead>`;

                tableHtml += `<tbody>`;
                tableRows.forEach((row, index) => {
                    const rowBg = index % 2 === 0 ? "rgba(255, 255, 255, 0.02)" : "rgba(255, 255, 255, 0.05)";
                    tableHtml += `<tr style="background: ${rowBg}; border-bottom: 1px solid rgba(255, 255, 255, 0.05);">`;
                    headers.forEach(header => {
                        let value = row[header];
                        if (typeof value === "number" && !Number.isInteger(value)) {
                            value = value.toFixed(2);
                        }
                        tableHtml += `<td style="padding: 10px 12px;">${value !== null ? value : "-"}</td>`;
                    });
                    tableHtml += `</tr>`;
                });
                tableHtml += `</tbody></table>`;

                tableContainer.innerHTML = tableHtml;
                contentDiv.appendChild(tableContainer);
            });
        }

        // --- NEW: ANALYTICAL PLOTLY CHARTS RENDERING ---
        if (data.charts && data.charts.length > 0) {
            data.charts.forEach(chartObj => {
                let chartData = chartObj.plotly_json;

                // Handle double serialization
                if (typeof chartData === "string") {
                    try {
                        chartData = JSON.parse(chartData);
                    } catch (e) {
                        console.error("Failed to parse plotly_json string inside charts array:", e);
                        chartData = null;
                    }
                }

                if (chartData && chartData.data && chartData.layout) {
                    const chartDiv = document.createElement("div");
                    const chartId = "plotly-chart-anal-" + Date.now() + Math.random().toString(36).substr(2, 5);
                    chartDiv.id = chartId;
                    chartDiv.style.width = "100%";
                    chartDiv.style.marginTop = "15px";
                    contentDiv.appendChild(chartDiv);

                    setTimeout(() => {
                        const layout = {
                            ...chartData.layout,
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#e2e8f0', ...chartData.layout?.font }
                        };
                        Plotly.newPlot(chartId, chartData.data, layout, { responsive: true });
                    }, 50);
                }
            });
        }

        // --- EXPLANATION CHART RENDERING (ML/SHAP) ---
        let chartData = data.explanation_chart;
        if (typeof chartData === "string") {
            try {
                chartData = JSON.parse(chartData);
            } catch (e) {
                console.error("Failed to parse explanation_chart string:", e);
                chartData = null;
            }
        }

        if (chartData && chartData.data && chartData.layout) {
            const chartDiv = document.createElement("div");
            const chartId = "plotly-chart-ml-" + Date.now();
            chartDiv.id = chartId;
            chartDiv.style.width = "100%";
            chartDiv.style.marginTop = "15px";
            contentDiv.appendChild(chartDiv);

            setTimeout(() => {
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

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}