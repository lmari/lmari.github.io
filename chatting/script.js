let interactiveReadingEnabled = false;
let currentParagraph = 0;

function showNextParagraph(paragraphs) {
    currentParagraph++;
    if (currentParagraph < paragraphs.length) {
        paragraphs[currentParagraph].style.display = "block";
    }
}

function handleInteractiveKeydown(event) {
    if (!interactiveReadingEnabled || event.key !== " ") {
        return;
    }

    event.preventDefault();
    showNextParagraph(document.querySelectorAll("p, pre"));
}

function handleInteractiveClick() {
    if (!interactiveReadingEnabled) {
        return;
    }

    showNextParagraph(document.querySelectorAll("p, pre"));
}

document.addEventListener("keydown", handleInteractiveKeydown);
document.addEventListener("click", handleInteractiveClick);

function mode() {
    const paragraphs = document.querySelectorAll("p, pre");
    const modeButton = document.getElementById("mode");
    const fullPageText = "passa alla lettura interattiva";
    const interactiveText = "(spazio o click per avanzare) passa alla pagina completa";

    modeButton.textContent = modeButton.textContent == fullPageText ? interactiveText : fullPageText;
    interactiveReadingEnabled = modeButton.textContent == interactiveText;

    if (interactiveReadingEnabled) {
        for (let i = 1; i < paragraphs.length; i++) {
            paragraphs[i].style.display = "none";
        }
        currentParagraph = 0;
    } else {
        for (let i = 1; i < paragraphs.length; i++) {
            paragraphs[i].style.display = "block";
        }
    }
}
