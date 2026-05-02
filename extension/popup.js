document.addEventListener('DOMContentLoaded', function() {
    const docsBtn = document.getElementById('docsBtn');
    if (docsBtn) {
        docsBtn.addEventListener('click', function() {
            window.open('http://localhost:8000/docs', '_blank');
        });
    }
});
