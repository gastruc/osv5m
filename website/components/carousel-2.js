const wrapper2 = document.querySelector(".wrapper2");
const carousel2 = document.querySelector(".carousel2");
const arrowBtns2 = document.querySelectorAll(".wrapper2 i");
const carousel2Childrens = [...carousel2.children];

let isdragging22 = false, isautoPlay22 = true, startX2, startScrollLeft2, timeoutId2;

// Get the number of card2s that can fit in the carousel2 at once
let card2PerView = 1;

// Scroll the carousel2 at appropriate position to hide first few duplicate card2s on Firefox
carousel2.classList.add("no-transition");
carousel2.scrollLeft = 0;
carousel2.classList.remove("no-transition");

// Add event listeners for the arrow buttons to scroll the carousel2 left and right
arrowBtns2.forEach(btn => {
    btn.addEventListener("click", () => {
        let firstCard = carousel2.querySelector(".card2");
        let firstCardWidth = firstCard.offsetWidth + parseInt(getComputedStyle(firstCard).marginRight);
        
        if (btn.id == "right" && carousel2.scrollLeft + carousel2.offsetWidth >= carousel2.scrollWidth) {
            carousel2.scrollLeft = 0;
        } else {
            carousel2.scrollLeft += btn.id == "left" ? -firstCardWidth : firstCardWidth;
        }
    });
});
const dragStart2 = (e) => {
    isdragging22 = true;
    carousel2.classList.add("dragging2");
    // Records the initial cursor and scroll position of the carousel2
    startX2 = e.pageX;
    startScrollLeft2 = carousel2.scrollLeft;
}

const dragging2 = (e) => {
    if(!isdragging22) return; // if isdragging22 is false return from here
    // Updates the scroll position of the carousel2 based on the cursor movement
    carousel2.scrollLeft = startScrollLeft2 - (e.pageX - startX2);
}

const dragStop2 = () => {
    isdragging22 = false;
    carousel2.classList.remove("dragging2");
}

const infiniteScroll2 = () => {
    // If the carousel2 is at the beginning, scroll to the end
    if(carousel2.scrollLeft === 0) {
        // carousel2.classList.add("no-transition");
        carousel2.scrollLeft = carousel2.scrollWidth;
        // carousel2.classList.remove("no-transition");
    }
    // If the carousel2 is at the end, scroll to the beginning
    else if(Math.ceil(carousel2.scrollLeft) === carousel2.scrollWidth) {
        // carousel2.classList.add("no-transition");
        carousel2.scrollLeft = 0;
        // carousel2.classList.remove("no-transition");
    }

    // Clear existing timeout & start autoPlay2 if mouse is not hovering over carousel2
    clearTimeout(timeoutId2);
    if(!wrapper2.matches(":hover")) autoPlay2();
}

const autoPlay2 = () => {
    if(window.innerWidth < 800 || !isautoPlay22) return; // Return if window is smaller than 800 or isautoPlay22 is false
    // autoPlay2 the carousel2 after every 2500 ms
    // timeoutId2 = setTimeout(() => {
    //     let firstCard = carousel2.querySelector(".card2");
    //     let firstCardWidth = firstCard.offsetWidth + parseInt(getComputedStyle(firstCard).marginRight);
    //     carousel2.scrollLeft += firstCardWidth;
    // }, 5000);
}
autoPlay2();

carousel2.addEventListener("mousedown", dragStart2);
carousel2.addEventListener("mousemove", dragging2);
document.addEventListener("mouseup", dragStop2);
carousel2.addEventListener("scroll", infiniteScroll2);
wrapper2.addEventListener("mouseenter", () => clearTimeout(timeoutId2));
wrapper2.addEventListener("mouseleave", autoPlay2);