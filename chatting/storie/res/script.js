function checkVisibility() {
    const sections = document.querySelectorAll('.section');
    const scrollPosition = window.scrollY + window.innerHeight;
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        if (scrollPosition >= sectionTop && scrollPosition <= sectionTop + sectionHeight) {
            section.classList.add('active-section');
        } else {
            section.classList.remove('active-section');
        }
    });
}

window.addEventListener('scroll', checkVisibility);

document.addEventListener('DOMContentLoaded', () => {
    checkVisibility();
});