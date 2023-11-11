const mobileMenu = document.getElementById("mobile-menu");

// Get the mobile menu button by its aria-controls attribute
const mobileMenuButton = document.querySelector("[aria-controls='mobile-menu']");

const checkboxContainer = document.getElementById("checkbox-container");

const plotType = document.getElementById("plot-type");
const modelType = document.getElementById('model-type')
const dataDropdownSelection = document.getElementById('dataset')

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


// Fetch features and create checkboxes
function createFeatureCheckboxes(features) {
    // Check if features is an array
    if (!Array.isArray(features)) {
        console.error('Expected an array of features, but got:', features);
        return;
    }

    checkboxContainer.innerHTML = ''; // Clear existing checkboxes
    features.forEach(feature => {
        if (feature === 'date') {
            return; // Skip date feature
        }
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
    console.log('Checkboxes created for features:', features);  // Debugging line
}


// Initial fetch of features when the page loads
fetch('/features', {
    method: 'GET'
})
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
})
.then(features => {
    createFeatureCheckboxes(features);  // Call the function here
})
.catch(error => console.error('Error:', error));



// When a new model is selected, fetch the new model's features
if (modelType) {
    modelType.addEventListener('change', () => {
        const model = modelType.value;

        // Send POST request with model type
        fetch('/features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({model: model}),
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error("Failed to fetch features");
            }
        })
        .then(data => {
            // Handle the fetched feature data
            console.log(`Features for ${model}:`, data);
            // Update your UI with the new features here
        })
        .catch(error => {
            console.error('Error fetching features:', error);
        });
    });
}


// When a new dataset is selected, fetch the new dataset's features and update the checkboxes
if (dataDropdownSelection){
    dataDropdownSelection.addEventListener('change', () => {
    const dataset = dataDropdownSelection.value;

    // Fetch new features based on the selected model
    fetch(`/features?data=${dataset}`)
        .then(response => response.json())
        .then(features => {
            if (checkboxContainer) {
                createFeatureCheckboxes(features);
            }
        })
        .catch(error => console.error('Error:', error));
    });
}

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

// Update the UI with model evaluation metrics
function updateUIWithMetrics(data) {
    document.getElementById('mse').textContent = 'MSE: ' + data.mse;
    document.getElementById('mae').textContent = 'MAE: ' + data.mae;
    document.getElementById('rmse').textContent = 'RMSE: ' + data.rmse;
    document.getElementById('mape').textContent = 'MAPE: ' + data.mape;
    document.getElementById('r2').textContent = 'R2: ' + data.r2;

    document.getElementById('actual-vs-predicted').src = data.actual_vs_predicted_plot;
    document.getElementById('feature-importance').src = data.feature_importance_plot || 'Model does not support feature importance';

    // Hide loading indicator and show the plots
    // document.getElementById('loading').classList.add('hidden');
    document.getElementById('actual-vs-predicted').classList.remove('hidden');
    document.getElementById('feature-importance').classList.remove('hidden');
}

// Predict
function predict() {
    console.log("PREDICTING!!");
    const model = modelType.value;

    // TODO: Show loading indicator
    // TODO: Make this dynamic by fetching the features for the selected model
    // Fetch features first
    fetch('/features', {
        method: 'GET',  // Specify the method
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Failed to fetch features");
        }
        return response.json();
    })
    .then(features => {
        console.log("Selected features:", features);  // Debugging line

        // Prepare data to send to the server
        const data = {
            model_type: model,
            selected_features: features  // Use the fetched features
        };

        // Send a POST request to the server to get the prediction
        return fetch('/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Failed to fetch prediction");
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        // Update UI with the received data
        updateUIWithMetrics(data);
    })
    .catch(error => {
        console.error("Error:", error);
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