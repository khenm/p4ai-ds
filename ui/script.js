document.addEventListener('DOMContentLoaded', () => {

    // Timestamp Update
    const updateTimestamp = () => {
        const now = new Date();
        const el = document.getElementById('timestamp');
        if (el) {
            el.textContent = now.toISOString().slice(0, 19).replace('T', ' ') + ' UTC';
        }
    };
    updateTimestamp();
    setInterval(updateTimestamp, 1000);

    // Tab Navigation Logic
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active classes
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab
            tab.classList.add('active');

            // Find matching content and show
            const targetId = tab.dataset.tab;
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    console.log("PetFinder Dashboard Initialized.");
});
