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


async function handleFileUpload(event) {
    event.preventDefault();
    event.stopPropagation();

    const files = event.target.files;
    if (!files || files.length === 0) return;

    const filesToUpload = Array.from(files).slice(0, 2);
    updateStatusBadge("Uploading...", "offline");
    addSystemMessage("Uploading and processing your dataset(s)... Please wait.");

    try {
        if (filesToUpload.length === 2) {
            // Both selected at once
            const primaryRes = await uploadFile(filesToUpload[0]);
            primaryDatasetId = primaryRes.dataset_id;
            addSystemMessage(`✅ Primary dataset loaded: <b>${primaryRes.filename}</b> (${primaryRes.row_count} rows).`);

            const predRes = await uploadFile(filesToUpload[1]);
            predictionDatasetId = predRes.dataset_id;
            addSystemMessage(`✅ Prediction dataset loaded: <b>${predRes.filename}</b> (${predRes.row_count} rows).`);
            updateStatusBadge("2 Datasets Active", "online");

        } else if (filesToUpload.length === 1) {
            if (!primaryDatasetId) {
                // First upload ever -> set as primary
                const primaryRes = await uploadFile(filesToUpload[0]);
                primaryDatasetId = primaryRes.dataset_id;
                addSystemMessage(`✅ Primary dataset loaded: <b>${primaryRes.filename}</b> (${primaryRes.row_count} rows).`);
                updateStatusBadge("Dataset Active", "online");
            } else {
                // Primary already exists -> set as prediction dataset!
                const predRes = await uploadFile(filesToUpload[0]);
                predictionDatasetId = predRes.dataset_id;
                addSystemMessage(`✅ Prediction dataset loaded: <b>${predRes.filename}</b> (${predRes.row_count} rows).`);
                updateStatusBadge("2 Datasets Active", "online");
            }
        }

        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.placeholder = "Ask a question about your dataset...";
        queryInput.focus();

    } catch (error) {
        console.error("Upload Error:", error);
        addSystemMessage(`❌ Upload failed: ${error.message}`);
    } finally {
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
 * Parses and displays AI outputs (text, predictions, tables, and Plotly charts)
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
        contentDiv.innerHTML = `<ul class="response-list">${listItems}</ul>`;
    } 
    // 4. Handle Objects (Charts, Tables, Predictions, Summaries)
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

        // --- DYNAMIC PREDICTIONS RENDERING ---
        if (data.predictions) {
            const predObj = data.predictions;
            const predContainer = document.createElement("div");
            predContainer.className = "predictions-container";

            let predHtml = "";

            // Display MAE Score if available
            if (predObj.mae_score !== undefined && predObj.mae_score !== null) {
                const maeVal = typeof predObj.mae_score === "number" ? predObj.mae_score.toFixed(4) : predObj.mae_score;
                predHtml += `<div class="mae-score-badge">
                    📊 MAE Score: <span class="mae-value">${maeVal}</span>
                </div>`;
            }

            // Display Message if present
            if (predObj.message) {
                predHtml += `<p class="prediction-message">${escapeHtml(predObj.message)}</p>`;
            }

            // Display Predicted Values inside a styled table
            if (predObj.predicted_values && predObj.predicted_values.length > 0) {
                predHtml += `<div class="prediction-title">Predicted Output Values (${predObj.predicted_values.length} rows):</div>`;
                predHtml += `<div class="prediction-table-wrapper">`;
                predHtml += `<table class="prediction-table">`;
                predHtml += `<thead><tr>
                    <th>Row #</th>
                    <th>Predicted Value</th>
                </tr></thead><tbody>`;

                predObj.predicted_values.forEach((val, idx) => {
                    const formattedVal = typeof val === "number" ? val.toFixed(4) : val;
                    predHtml += `<tr>
                        <td class="row-index">${idx + 1}</td>
                        <td class="pred-val">${formattedVal}</td>
                    </tr>`;
                });

                predHtml += `</tbody></table></div>`;
            }

            predContainer.innerHTML = predHtml;
            contentDiv.appendChild(predContainer);
        }

        // --- DYNAMIC TABLE RENDERING ---
        if (data.tables && data.tables.length > 0) {
            data.tables.forEach(tableObj => {
                const tableRows = tableObj.data;
                if (!tableRows || tableRows.length === 0) return;

                const headers = Object.keys(tableRows[0]);
                const tableContainer = document.createElement("div");
                tableContainer.className = "table-container";

                let tableHtml = `<table class="analysis-table">`;
                
                tableHtml += `<thead><tr>`;
                headers.forEach(header => {
                    const displayHeader = header.replace(/_/g, " ").toUpperCase();
                    tableHtml += `<th>${displayHeader}</th>`;
                });
                tableHtml += `</tr></thead>`;

                tableHtml += `<tbody>`;
                tableRows.forEach((row, index) => {
                    tableHtml += `<tr>`;
                    headers.forEach(header => {
                        let value = row[header];
                        if (typeof value === "number" && !Number.isInteger(value)) {
                            value = value.toFixed(2);
                        }
                        tableHtml += `<td>${value !== null ? value : "-"}</td>`;
                    });
                    tableHtml += `</tr>`;
                });
                tableHtml += `</tbody></table>`;

                tableContainer.innerHTML = tableHtml;
                contentDiv.appendChild(tableContainer);
            });
        }

        // --- ANALYTICAL PLOTLY CHARTS RENDERING ---
        if (data.charts && data.charts.length > 0) {
            data.charts.forEach(chartObj => {
                let chartData = chartObj.plotly_json;

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
                    chartDiv.className = "plotly-chart-container";
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
            chartDiv.className = "plotly-chart-container";
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
            // ✅ NEW: Narrative box directly beneath the SHAP chart
            if (data.explanation_text) {
                const narrativeBox = document.createElement("div");
                narrativeBox.className = "message bot-message chart-narrative-box";
                narrativeBox.innerHTML = `
                    <div class="chart-narrative-label">💡 What this chart means</div>
                    <p>${escapeHtml(data.explanation_text).replace(/\n/g, "<br>")}</p>
                `;
                contentDiv.appendChild(narrativeBox);
            }
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