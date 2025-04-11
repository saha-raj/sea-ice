document.addEventListener('DOMContentLoaded', function() {
    initStoryCarousel();
});

function initStoryCarousel() {
    const track = document.querySelector('.stories-track');
    const cards = document.querySelectorAll('.story-card');
    const prevButton = document.querySelector('.carousel-nav.prev');
    const nextButton = document.querySelector('.carousel-nav.next');
    
    if (!track || cards.length === 0) {
        return;
    }
    
    let currentIndex = 0;
    let cardWidth = 0;
    let visibleCards = 3; // Default for medium screens
    
    // Function to update carousel state
    function updateCarousel() {
        // Determine how many cards to show based on screen width
        if (window.innerWidth >= 1100) {
            visibleCards = 4;
        } else if (window.innerWidth >= 900) {
            visibleCards = 3;
        } else if (window.innerWidth >= 600) {
            visibleCards = 2;
        } else {
            visibleCards = 1;
        }
        
        // Calculate card width (including margins)
        const carousel = document.querySelector('.stories-carousel');
        if (carousel) {
            const carouselWidth = carousel.offsetWidth;
            cardWidth = carouselWidth / visibleCards;
            
            // Update card width in the DOM
            cards.forEach(card => {
                card.style.width = `calc(${100 / visibleCards}% - 20px)`;
            });
        }
        
        // Update track position
        moveToIndex(currentIndex);
    }
    
    // Function to move to a specific index
    function moveToIndex(index) {
        const maxIndex = Math.max(0, cards.length - visibleCards);
        currentIndex = Math.min(Math.max(0, index), maxIndex);
        
        // Calculate the translation
        const translateX = -currentIndex * cardWidth;
        track.style.transform = `translateX(${translateX}px)`;
        
        // Update button states
        prevButton.disabled = currentIndex === 0;
        prevButton.style.opacity = currentIndex === 0 ? '0.5' : '1';
        nextButton.disabled = currentIndex >= maxIndex;
        nextButton.style.opacity = currentIndex >= maxIndex ? '0.5' : '1';
    }
    
    // Initialize carousel
    updateCarousel();
    
    // Add event listeners for navigation
    prevButton.addEventListener('click', () => moveToIndex(currentIndex - 1));
    nextButton.addEventListener('click', () => moveToIndex(currentIndex + 1));
    
    // Handle keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowLeft') {
            moveToIndex(currentIndex - 1);
        } else if (e.key === 'ArrowRight') {
            moveToIndex(currentIndex + 1);
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', debounce(updateCarousel, 250));
    
    // Add touch swipe functionality
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
        if (touchStartX - touchEndX > 50) {
            // Swipe left, move to next
            moveToIndex(currentIndex + 1);
        } else if (touchEndX - touchStartX > 50) {
            // Swipe right, move to previous
            moveToIndex(currentIndex - 1);
        }
    }
}

// Utility function to debounce resize events
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
} 