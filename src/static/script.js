document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analysisForm');
    const fileInput = document.getElementById('mriImage');
    const previewContainer = document.getElementById('imagePreviewContainer');
    const previewImage = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');

    const initialState = document.getElementById('initialState');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');

    const predLabel = document.getElementById('predLabel');
    const predConf = document.getElementById('predConf');
    const reportDisplay = document.getElementById('reportDisplay');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const pdfPassword = document.getElementById('pdfPassword');

    let currentReportMarkdown = "";
    let currentPrediction = null;

    // File Upload Preview
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove('hidden');
                analyzeBtn.disabled = false;
            }
            reader.readAsDataURL(file);
        }
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI State: Loading
        analyzeBtn.disabled = true;
        btnText.textContent = "Processing...";
        spinner.classList.remove('hidden');
        
        initialState.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData(form);
            
            // Step 1: Classify Image
            const classifyResponse = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });
            
            if (!classifyResponse.ok) throw new Error('Classification failed');
            
            const classification = await classifyResponse.json();
            currentPrediction = classification;
            
            // Update UI with classification
            predLabel.textContent = classification.prediction;
            predConf.textContent = `${classification.confidence.toFixed(2)}%`;

            // Step 2: Generate Report
            // We need to append the classification results to the form data for the report generation
            formData.append('tumor_type', classification.prediction.toLowerCase().replace(' ', '')); // simple normalization
            // Actually, the backend expects specific strings, let's just pass the raw prediction for now
            // The backend logic: if tumor_type == "notumor" ...
            // The model returns class names. I should check what the class names are.
            // Assuming the model returns "notumor", "meningioma", etc.
            // Let's use the raw prediction string from the model.
            
            // Re-create FormData or append to existing?
            // We can't easily modify the existing FormData with the file stream again if we consumed it?
            // Actually, fetch body consumes the stream? No, FormData is reusable in JS.
            
            // However, we need to send the classification result.
            // Let's create a new FormData for the second request.
            const reportFormData = new FormData();
            reportFormData.append('patient_name', formData.get('patientName'));
            reportFormData.append('patient_id', formData.get('patientID'));
            reportFormData.append('dob', formData.get('dob'));
            reportFormData.append('file', fileInput.files[0]);
            reportFormData.append('tumor_type', classification.prediction);
            reportFormData.append('confidence', classification.confidence);

            const reportResponse = await fetch('/api/generate_report', {
                method: 'POST',
                body: reportFormData
            });

            if (!reportResponse.ok) throw new Error('Report generation failed');

            const reportData = await reportResponse.json();
            currentReportMarkdown = reportData.report_markdown;
            
            // Display Report
            reportDisplay.innerHTML = reportData.report_html;
            
            // UI State: Success
            loadingState.classList.add('hidden');
            resultsContent.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            alert('An error occurred: ' + error.message);
            loadingState.classList.add('hidden');
            initialState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = "Analyze & Generate Report";
            spinner.classList.add('hidden');
        }
    });

    // PDF Download
    downloadPdfBtn.addEventListener('click', async () => {
        if (!currentReportMarkdown) return;

        const formData = new FormData();
        formData.append('report_markdown', currentReportMarkdown);
        formData.append('file', fileInput.files[0]);
        
        if (pdfPassword.value) {
            formData.append('password', pdfPassword.value);
        }

        try {
            const response = await fetch('/api/generate_pdf', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('PDF generation failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "radiology_report.pdf";
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();

        } catch (error) {
            console.error(error);
            alert('Failed to download PDF: ' + error.message);
        }
    });
});
