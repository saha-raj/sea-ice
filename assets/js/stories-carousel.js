document.addEventListener('DOMContentLoaded', function() {
    initStoryCarousel();
});

function initStoryCarousel() {
    const track = document.querySelector('.stories-track');
    const cardsNodeList = track ? track.querySelectorAll('.story-card') : null;
    
    if (!track || !cardsNodeList || cardsNodeList.length === 0) {
        return;
    }
    
    const cardsArray = Array.from(cardsNodeList);
    
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    const shuffledCards = shuffleArray(cardsArray);
    
    track.innerHTML = '';
    
    shuffledCards.forEach(card => {
        track.appendChild(card);
    });
    
    const cards = track.querySelectorAll('.story-card');
    const prevButton = document.querySelector('.carousel-nav.prev');
    const nextButton = document.querySelector('.carousel-nav.next');
    
    if (!prevButton || !nextButton) {
        console.error("Carousel navigation buttons not found.");
        return;
    }
    
    let currentIndex = 0;
    let cardWidth = 0;
    let visibleCards = 3; // Default for medium screens
    
    function updateCarousel() {
        if (window.innerWidth >= 1100) {
            visibleCards = 4;
        } else if (window.innerWidth >= 900) {
            visibleCards = 3;
        } else if (window.innerWidth >= 600) {
            visibleCards = 2;
        } else {
            visibleCards = 1;
        }
        
        const carousel = document.querySelector('.stories-carousel');
        if (carousel) {
            const carouselWidth = carousel.offsetWidth;
            const actualVisibleCards = Math.max(1, visibleCards);
            cardWidth = carouselWidth / actualVisibleCards;
            
            cards.forEach(card => {
                card.style.width = `calc(${100 / actualVisibleCards}% - 20px)`;
                card.style.marginLeft = '10px';
                card.style.marginRight = '10px';
            });
        } else {
            console.error("'.stories-carousel' container not found for width calculation.");
            return;
        }
        
        moveToIndex(currentIndex);
    }
    
    function moveToIndex(index) {
        if (cards.length === 0) return;
        
        const maxIndex = Math.max(0, cards.length - visibleCards);
        currentIndex = Math.min(Math.max(0, index), maxIndex);
        
        const spacePerCard = cardWidth;
        const translateX = -currentIndex * spacePerCard;
        
        track.style.transform = `translateX(${translateX}px)`;
        
        prevButton.disabled = currentIndex === 0;
        prevButton.style.opacity = currentIndex === 0 ? '0.5' : '1';
        nextButton.disabled = currentIndex >= maxIndex;
        nextButton.style.opacity = currentIndex >= maxIndex ? '0.5' : '1';
    }
    
    updateCarousel();
    
    prevButton.addEventListener('click', () => moveToIndex(currentIndex - 1));
    nextButton.addEventListener('click', () => moveToIndex(currentIndex + 1));
    
    document.addEventListener('keydown', function(e) {
        if (document.activeElement && document.activeElement.closest('.stories-carousel-container')) {
            if (e.key === 'ArrowLeft') {
                moveToIndex(currentIndex - 1);
            } else if (e.key === 'ArrowRight') {
                moveToIndex(currentIndex + 1);
            }
        }
    });
    
    window.addEventListener('resize', debounce(updateCarousel, 250));
    
    let touchStartX = 0;
    let touchEndX = 0;
    
    track.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });
    
    track.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, { passive: true });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        if (touchStartX - touchEndX > swipeThreshold) {
            moveToIndex(currentIndex + 1);
        } else if (touchEndX - touchStartX > swipeThreshold) {
            moveToIndex(currentIndex - 1);
        }
        touchStartX = 0;
        touchEndX = 0;
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
} 