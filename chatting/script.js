function mode() {
    const paragraphs = document.querySelectorAll("p");
    x = document.getElementById("mode")
    x1 = "passa alla lettura interattiva"
    x2 = "(spazio o click per avanzare) passa alla pagina completa"
    x.textContent = x.textContent == x1 ? x2 : x1

    if (x.textContent == x2) {
        for (let i = 1; i < paragraphs.length; i++) {
            paragraphs[i].style.display = "none"
        }
        let currentParagraph = 0

        function showNext() {
            currentParagraph++;
            if (currentParagraph < paragraphs.length) {
                paragraphs[currentParagraph].style.display = "block"
            }
        }

        document.addEventListener("keydown", function(event) {
            if (event.key === " ") {
                showNext()
            }
        })

        document.addEventListener("click", function() {
            showNext();
        })

    } else {
        for (let i = 1; i < paragraphs.length; i++) {
            paragraphs[i].style.display = "block";
        }
    }
            
}
