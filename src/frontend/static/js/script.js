
document.addEventListener("DOMContentLoaded", function () {
    const customersElement = document.getElementById("customers");
    const productsElement = document.getElementById("products");
    const statusElement = document.getElementById("status");
    const loadButton = document.getElementById("loadStats");
    const chartElement = document.getElementById("purchaseResults");

    // ==========================================
    // 1. LOAD DASHBOARD STATISTICS
    // ==========================================
    async function loadAnalytics() {
        if (statusElement) {
            statusElement.textContent = "Loading analytics...";
        }

        try {
            const response = await fetch("/stats");

            if (!response.ok) {
                throw new Error("Failed to load dashboard statistics.");
            }

            const data = await response.json();

            if (customersElement) {
                customersElement.textContent = data.total_customers;
            }

            if (productsElement) {
                productsElement.textContent = data.total_products;
            }

            if (statusElement) {
                statusElement.textContent = "Analytics loaded successfully!";
            }

            await loadCustomerChart();

        } catch (error) {
            console.error("Analytics error:", error);

            if (statusElement) {
                statusElement.textContent =
                    "Unable to load analytics. Please check the backend.";
            }
        }
    }

    // ==========================================
    // 2. CUSTOMER PURCHASE ANALYSIS
    // ==========================================
    async function loadCustomerChart() {
        if (!chartElement) {
            console.warn('Element "purchaseResults" was not found.');
            return;
        }

        try {
            const response = await fetch("/customer-analysis");

            if (!response.ok) {
                throw new Error("Failed to load customer analysis.");
            }

            const data = await response.json();

            const customers = data.customers || [];
            const orders = data.total_orders || [];

            if (customers.length === 0 || orders.length === 0) {
                chartElement.textContent =
                    "No customer purchase data is available.";
                return;
            }

            const count = Math.min(
                10,
                customers.length,
                orders.length
            );

            chartElement.innerHTML = "";

            const heading = document.createElement("h3");
            heading.textContent = "Customer Purchase Analysis";
            chartElement.appendChild(heading);

            for (let i = 0; i < count; i++) {
                const row = document.createElement("p");

                row.textContent =
                    "Customer " + customers[i] +
                    " — Orders: " + orders[i];

                chartElement.appendChild(row);
            }

        } catch (error) {
            console.error("Customer chart error:", error);

            chartElement.textContent =
                "Unable to display customer purchase analysis.";
        }
    }

    // ==========================================
    // 3. RECOMMENDATION EVALUATION CHART
    // ==========================================
    async function loadEvaluation() {
    try {
        const response = await fetch("/evaluation");
        if (!response.ok) {
            throw new Error("Failed to load evaluation");
        }

        const data = await response.json();

        console.log("Evaluation data:", data);

        const model5 = data.item_based_cosine.hit_rate_at_5 * 100;
        const base5 = data.popularity_baseline.hit_rate_at_5 * 100;
        const model10 = data.item_based_cosine.hit_rate_at_10 * 100;
        const base10 = data.popularity_baseline.hit_rate_at_10 * 100;

        updateBar("chart-model-5", model5);
        updateBar("chart-base-5", base5);
        updateBar("chart-model-10", model10);
        updateBar("chart-base-10", base10);

    } catch (error) {
        console.error("Evaluation error:", error);
    }
}

function updateBar(id, value) {
    const bar = document.getElementById(id);

    if (bar) {
        bar.style.width = value + "%";
        bar.textContent = value.toFixed(2) + "%";
    }
}

    // ==========================================
    // 4. LOAD ALL DATA WHEN DASHBOARD OPENS
    // ==========================================
    if (loadButton) {
        loadButton.addEventListener("click", function () {
            loadAnalytics();
            loadEvaluationChart();
        });
    }

    loadAnalytics();
    loadEvaluationChart();

    // Auto-load rfm.csv from the same folder
fetch("rfm.csv")
  .then(function(r){
    if(!r.ok) throw new Error("rfm.csv not found");
    return r.blob();
  })
  .then(function(b){ loadRFM(new File([b], "rfm.csv")); })
  .catch(function(){ /* file not found: user can upload manually */ });

});
