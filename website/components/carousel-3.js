const wrapper3 = document.querySelector(".wrapper3");
const carousel3 = document.querySelector(".carousel3");
const arrowBtns3 = document.querySelectorAll(".wrapper3 i");
const carousel3Childrens = [...carousel3.children];

let isdragging33 = false, isautoPlay33 = true, startX3, startScrollLeft3, timeoutId3;

// Get the number of card3s that can fit in the carousel3 at once
let card3PerView = 1;

// Scroll the carousel3 at appropriate position to hide first few duplicate card3s on Firefox
carousel3.classList.add("no-transition");
carousel3.scrollLeft = 0;
carousel3.classList.remove("no-transition");

// Add event listeners for the arrow buttons to scroll the carousel3 left and right
arrowBtns3.forEach(btn => {
    btn.addEventListener("click", () => {
        let firstCard = carousel3.querySelector(".card3");
        let firstCardWidth = firstCard.offsetWidth + parseInt(getComputedStyle(firstCard).marginRight);
        
        if (btn.id == "right" && carousel3.scrollLeft + carousel3.offsetWidth >= carousel3.scrollWidth) {
            carousel3.scrollLeft = 0;
        } else {
            carousel3.scrollLeft += btn.id == "left" ? -firstCardWidth : firstCardWidth;
        }
    });
});
const dragStart3 = (e) => {
    isdragging33 = true;
    carousel3.classList.add("dragging3");
    // Records the initial cursor and scroll position of the carousel3
    startX3 = e.pageX;
    startScrollLeft3 = carousel3.scrollLeft;
}

const dragging3 = (e) => {
    if(!isdragging33) return; // if isdragging33 is false return from here
    // Updates the scroll position of the carousel3 based on the cursor movement
    carousel3.scrollLeft = startScrollLeft3 - (e.pageX - startX3);
}

const dragStop3 = () => {
    isdragging33 = false;
    carousel3.classList.remove("dragging3");
}

const infiniteScroll3 = () => {
    // If the carousel3 is at the beginning, scroll to the end
    if(carousel3.scrollLeft === 0) {
        // carousel3.classList.add("no-transition");
        carousel3.scrollLeft = carousel3.scrollWidth;
        // carousel3.classList.remove("no-transition");
    }
    // If the carousel3 is at the end, scroll to the beginning
    else if(Math.ceil(carousel3.scrollLeft) === carousel3.scrollWidth) {
        // carousel3.classList.add("no-transition");
        carousel3.scrollLeft = 0;
        // carousel3.classList.remove("no-transition");
    }

    // Clear existing timeout & start autoPlay3 if mouse is not hovering over carousel3
    clearTimeout(timeoutId3);
    if(!wrapper3.matches(":hover")) autoPlay3();
}

const autoPlay3 = () => {
    if(window.innerWidth < 800 || !isautoPlay33) return; // Return if window is smaller than 800 or isautoPlay33 is false
    // autoPlay3 the carousel3 after every 3500 ms
    // timeoutId3 = setTimeout(() => {
    //     let firstCard = carousel3.querySelector(".card3");
    //     let firstCardWidth = firstCard.offsetWidth + parseInt(getComputedStyle(firstCard).marginRight);
    //     carousel3.scrollLeft += firstCardWidth;
    // }, 5000);
}
autoPlay3();

carousel3.addEventListener("mousedown", dragStart3);
carousel3.addEventListener("mousemove", dragging3);
document.addEventListener("mouseup", dragStop3);
carousel3.addEventListener("scroll", infiniteScroll3);
wrapper3.addEventListener("mouseenter", () => clearTimeout(timeoutId3));
wrapper3.addEventListener("mouseleave", autoPlay3);