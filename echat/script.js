function mode() {
    const paragraphs = document.querySelectorAll("p, pre");
    x = document.getElementById("mode")
    x1 = "switch to interactive page"
    x2 = "(space or click to move on) switch to full page"
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