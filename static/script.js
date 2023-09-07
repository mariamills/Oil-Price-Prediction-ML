const mobileMenu = document.getElementById("mobile-menu");

// Get the mobile menu button by its aria-controls attribute
const mobileMenuButton = document.querySelector("[aria-controls='mobile-menu']");

// Listen for click events on the mobile menu button
mobileMenuButton.addEventListener("click", function() {
    // Toggle the visibility of the mobile menu
    if (mobileMenu.style.display === "block") {
        mobileMenu.style.display = "none";
    } else {
        mobileMenu.style.display = "block";
    }
});

// Window resize event to hide the mobile menu for larger screens
window.addEventListener("resize", () => {
  if (window.innerWidth >= 640) {  // 640px is the breakpoint for sm: classes in Tailwind by default
    mobileMenu.style.display = "none";
  }
});