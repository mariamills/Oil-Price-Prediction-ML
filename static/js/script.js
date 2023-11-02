const mobileMenu = document.getElementById("mobile-menu");

// Get the mobile menu button by its aria-controls attribute
const mobileMenuButton = document.querySelector("[aria-controls='mobile-menu']");

const plotType = document.getElementById("plot-type");

// Listen for click events on the mobile menu button
if (mobileMenuButton) {
    mobileMenuButton.addEventListener("click", function () {
        // Toggle the visibility of the mobile menu
        if (mobileMenu.style.display === "block") {
            mobileMenu.style.display = "none";
        } else {
            mobileMenu.style.display = "block";
        }
    });
}

// Get selected features
function getSelectedFeatures() {
    const checkboxes = document.querySelectorAll('input[name="feature"]:checked');
    let selected = [];
    checkboxes.forEach((checkbox) => {
        selected.push(checkbox.value);
    });
    console.log("Selected features:", selected);  // Debugging line
    return selected;
}

// Add checkboxes for feature selection
fetch('/features')
    .then(response => response.json())
    .then(data => {
        // Create a checkbox for each feature
        const checkboxContainer = document.getElementById("checkbox-container");

        if (checkboxContainer) {
            data.forEach(feature => {
                const wrapper = document.createElement("div");
                wrapper.classList.add("flex", "items-center", "relative", "pb-4", "pt-3.5");

                const checkboxWrapper = document.createElement("div");
                checkboxWrapper.classList.add("ml-3", "flex", "h-6", "items-center")

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = feature;
                checkbox.name = "feature";
                checkbox.value = feature;
                checkbox.checked = true;
                checkbox.classList.add("h-4", "w-4", "rounded", "border-gray-300", "text-indigo-600", "focus:ring-indigo-600")

                const labelWrapper = document.createElement("div");
                labelWrapper.classList.add("min-w-0", "flex-1", "text-sm", "leading-6")
                const label = document.createElement("label");
                label.htmlFor = feature;
                label.textContent = feature;
                label.classList.add("font-medium", "text-gray-900", "pl-1");

                const span = document.createElement("span");
                span.classList.add("sr-only");
                span.textContent = feature;

                checkboxWrapper.appendChild(checkbox);
                labelWrapper.appendChild(label);
                labelWrapper.appendChild(span);
                wrapper.appendChild(checkboxWrapper);
                wrapper.appendChild(labelWrapper);
                checkboxContainer.appendChild(wrapper);
            });
        }

        // Attach event listener after checkboxes are created
        document.querySelectorAll('input[name="feature"]').forEach((checkbox) => {
            checkbox.addEventListener('change', () => {
                console.log("Checkbox changed");  // Debugging line
                const selected_features = getSelectedFeatures();
                fetch('/plot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ selected_features: selected_features, plot_type: plotType.value }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('plot').src = data.plot_url;
                    document.getElementById('legend').src = data.legend_url;
                });
            });
        });
    });

// Plot type selection
if (plotType) {
    plotType.addEventListener('change', () => {
        console.log("Plot type changed");  // Debugging line
        const selected_features = getSelectedFeatures();
        fetch('/plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({selected_features: selected_features, plot_type: plotType.value}),
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error("Failed to fetch plot");
                }
            })
            .then(data => {
                document.getElementById('plot').src = data.plot_url;
                document.getElementById('legend').src = data.legend_url;
            })
            .catch(error => {
                console.error("Error fetching plot:", error);
            });
    });
}


function fetchAndPlotData() {
    const selected_features = getSelectedFeatures();
    fetch('/plot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ selected_features: selected_features, plot_type: plotType.value }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('plot').src = data.plot_url;
        document.getElementById('legend').src = data.legend_url;
    });
}

// Show/hide the features section
function featureToggle(){
    const features = document.getElementById("features");
    features.classList.toggle("hidden");
}

// Uncheck all checkboxes
function uncheckAllFeatures() {
    document.querySelectorAll('input[name="feature"]').forEach((checkbox) => {
        checkbox.checked = false;
    });
}

function displayResults(data) {
    document.getElementById('mse').textContent = data.mse;
    document.getElementById('mae').textContent = data.mae;
    document.getElementById('rmse').textContent = data.rmse;

    let actualVsPredictedPlotImg = document.getElementById('actual-vs-predicted-plot');
    actualVsPredictedPlotImg.src = data.actual_vs_predicted_plot;

    let featureImportancePlotImg = document.getElementById('feature-importance-plot');
    featureImportancePlotImg.src = data.feature_importance_plot;
}

// Predict
// Predict
function predict() {
    console.log("PREDICTING!!");
    const modelType = document.getElementById('model-type').value;
    const selectedFeatures = getSelectedFeatures();  // get selected features using your existing function
    console.log("Selected features:", selectedFeatures);  // Debugging line

    // Prepare data to send to the server
    const data = {
        model_type: modelType,
        selected_features: selectedFeatures  // include selected features in data
    };

    // Send a POST request to the server to get the prediction
    fetch('/predict_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error("Failed to fetch prediction");
        }
    })
    .then(data => {
        console.log('Success:', data);
        document.getElementById('mse').textContent = 'MSE: ' + data.mse;
        document.getElementById('mae').textContent = 'MAE: ' + data.mae;
        document.getElementById('rmse').textContent = 'RMSE: ' + data.rmse;
        document.getElementById('actual-vs-predicted').src = data.actual_vs_predicted_plot;
        document.getElementById('feature-importance').src = data.feature_importance_plot;
        // Hide loading indicator and show the plots
        // document.getElementById('loading').classList.add('hidden');
        document.getElementById('actual-vs-predicted').classList.remove('hidden');
        document.getElementById('feature-importance').classList.remove('hidden');
    })
    .catch(error => {
        console.error("Error fetching prediction:", error);
    });
}


// Check all checkboxes
function checkAllFeatures() {
    document.querySelectorAll('input[name="feature"]').forEach((checkbox) => {
        checkbox.checked = true;
    });
}


// Window resize event to hide the mobile menu for larger screens
window.addEventListener("resize", () => {
  if (window.innerWidth >= 640) {  // 640px is the breakpoint for sm: classes in Tailwind by default
    mobileMenu.style.display = "none";
  }
});