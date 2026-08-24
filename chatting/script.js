let interactiveReadingEnabled = false;
let currentParagraph = 0;

function getInteractiveParagraphs() {
    return document.querySelectorAll("p, pre");
}

function showNextParagraph() {
    const paragraphs = getInteractiveParagraphs();
    currentParagraph++;

    if (currentParagraph < paragraphs.length) {
        paragraphs[currentParagraph].style.display = "block";
        paragraphs[currentParagraph].scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function handleInteractiveKeydown(event) {
    if (!interactiveReadingEnabled || event.key !== " ") {
        return;
    }

    const interactiveElement = event.target.closest("a, button, input, textarea, select");
    if (interactiveElement) {
        return;
    }

    event.preventDefault();
    showNextParagraph();
}

function handleInteractiveClick(event) {
    if (!interactiveReadingEnabled) {
        return;
    }

    if (event.target.closest("#mode, a, button, input, textarea, select")) {
        return;
    }

    showNextParagraph();
}

document.addEventListener("keydown", handleInteractiveKeydown);
document.addEventListener("click", handleInteractiveClick);

function mode() {
    const paragraphs = getInteractiveParagraphs();
    const modeButton = document.getElementById("mode");
    const fullPageText = "passa alla lettura interattiva";
    const interactiveText = "(spazio o click per avanzare) passa alla pagina completa";

    interactiveReadingEnabled = !interactiveReadingEnabled;
    modeButton.textContent = interactiveReadingEnabled ? interactiveText : fullPageText;

    paragraphs.forEach((paragraph, index) => {
        paragraph.style.display = !interactiveReadingEnabled || index === 0 ? "block" : "none";
    });

    currentParagraph = 0;
}
