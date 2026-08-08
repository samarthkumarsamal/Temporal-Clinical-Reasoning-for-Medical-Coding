/* =========================================================
   Temporal Clinical Reasoning for Medical Coding
   main.js

   Features:
   1. Sticky navigation
   2. Scroll progress indicator
   3. Mobile navigation
   4. Scroll reveal animations
   5. Active navigation highlighting
   6. Dynamic moving neural-network background
   7. Pointer interaction
   8. Reduced-motion accessibility
   ========================================================= */

(() => {
    "use strict";


    // =====================================================
    // 1. CHECK REDUCED MOTION PREFERENCE
    // =====================================================

    const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;


    // =====================================================
    // 2. STICKY NAVIGATION AND SCROLL PROGRESS
    // =====================================================

    const header = document.querySelector(".site-header");

    const progress = document.getElementById(
        "scroll-progress"
    );


    function updateScrollUI() {

        const scrollTop =
            window.scrollY ||
            document.documentElement.scrollTop;


        const scrollHeight =
            document.documentElement.scrollHeight -
            window.innerHeight;


        // Add background to navbar after scrolling
        if (header) {

            header.classList.toggle(
                "scrolled",
                scrollTop > 16
            );

        }


        // Update top scroll progress bar
        if (progress) {

            const ratio =
                scrollHeight > 0
                    ? scrollTop / scrollHeight
                    : 0;


            const percentage =
                Math.min(
                    ratio * 100,
                    100
                );


            progress.style.width =
                `${percentage}%`;

        }

    }


    window.addEventListener(
        "scroll",
        updateScrollUI,
        {
            passive: true
        }
    );


    updateScrollUI();


    // =====================================================
    // 3. MOBILE NAVIGATION
    // =====================================================

    const navToggle =
        document.getElementById(
            "nav-toggle"
        );


    const navLinks =
        document.getElementById(
            "nav-links"
        );


    if (navToggle && navLinks) {

        navToggle.addEventListener(
            "click",
            () => {

                const isOpen =
                    navLinks.classList.toggle(
                        "open"
                    );


                navToggle.setAttribute(
                    "aria-expanded",
                    String(isOpen)
                );

            }
        );


        // Close mobile menu after clicking a link
        navLinks
            .querySelectorAll("a")
            .forEach((link) => {

                link.addEventListener(
                    "click",
                    () => {

                        navLinks.classList.remove(
                            "open"
                        );


                        navToggle.setAttribute(
                            "aria-expanded",
                            "false"
                        );

                    }
                );

            });

    }


    // =====================================================
    // 4. SCROLL REVEAL ANIMATION
    // =====================================================

    const revealElements =
        document.querySelectorAll(
            ".reveal"
        );


    if (
        reducedMotion ||
        !("IntersectionObserver" in window)
    ) {

        revealElements.forEach(
            (element) => {

                element.classList.add(
                    "is-visible"
                );

            }
        );

    }

    else {

        const revealObserver =
            new IntersectionObserver(

                (entries, observer) => {

                    entries.forEach(
                        (entry) => {

                            if (
                                entry.isIntersecting
                            ) {

                                entry.target
                                    .classList.add(
                                        "is-visible"
                                    );


                                observer.unobserve(
                                    entry.target
                                );

                            }

                        }
                    );

                },

                {
                    threshold: 0.12,
                    rootMargin:
                        "0px 0px -30px 0px"
                }

            );


        revealElements.forEach(
            (element, index) => {

                // Small stagger animation
                const delay =
                    Math.min(
                        index % 4,
                        3
                    ) * 70;


                element.style.transitionDelay =
                    `${delay}ms`;


                revealObserver.observe(
                    element
                );

            }
        );

    }


    // =====================================================
    // 5. ACTIVE NAVIGATION SECTION
    // =====================================================

    const sections =
        [
            ...document.querySelectorAll(
                "main section[id]"
            )
        ];


    const anchorLinks =
        [
            ...document.querySelectorAll(
                ".nav-links a"
            )
        ];


    if (
        "IntersectionObserver" in window
    ) {

        const sectionObserver =
            new IntersectionObserver(

                (entries) => {

                    const visible =
                        entries
                            .filter(
                                (entry) =>
                                    entry.isIntersecting
                            )
                            .sort(
                                (a, b) =>
                                    b.intersectionRatio -
                                    a.intersectionRatio
                            )[0];


                    if (!visible) {
                        return;
                    }


                    anchorLinks.forEach(
                        (link) => {

                            const target =
                                link.getAttribute(
                                    "href"
                                );


                            link.classList.toggle(

                                "active",

                                target ===
                                    `#${visible.target.id}`

                            );

                        }
                    );

                },

                {
                    rootMargin:
                        "-25% 0px -60% 0px",

                    threshold: [
                        0.05,
                        0.2,
                        0.4
                    ]
                }

            );


        sections.forEach(
            (section) => {

                sectionObserver.observe(
                    section
                );

            }
        );

    }


    // =====================================================
    // 6. DYNAMIC PROFESSIONAL BACKGROUND
    // =====================================================

    const canvas =
        document.getElementById(
            "network-canvas"
        );


    // Stop here when canvas does not exist
    // or reduced motion is enabled
    if (
        !canvas ||
        reducedMotion
    ) {
        return;
    }


    const ctx =
        canvas.getContext(
            "2d",
            {
                alpha: true
            }
        );


    if (!ctx) {
        return;
    }


    let width = 0;

    let height = 0;

    let dpr = 1;

    let points = [];

    let animationFrame = null;


    // Mouse / pointer location
    const pointer = {

        x: null,
        y: null

    };


    // =====================================================
    // 7. PARTICLE CLASS
    // =====================================================

    class Point {

        constructor() {

            this.reset(true);

        }


        reset(initial = false) {

            // Random horizontal position
            this.x =
                Math.random() *
                width;


            // Start particles randomly during initial load
            // otherwise start below the screen
            this.y =
                initial
                    ? Math.random() *
                      height
                    : height + 20;


            // Slow horizontal movement
            this.vx =
                (
                    Math.random() -
                    0.5
                ) * 0.18;


            // Slow upward movement
            this.vy =
                -(
                    0.06 +
                    Math.random() *
                    0.14
                );


            // Particle size
            this.radius =
                0.55 +
                Math.random() *
                1.25;


            // Particle opacity
            this.alpha =
                0.14 +
                Math.random() *
                0.34;


            // Used for subtle wave movement
            this.phase =
                Math.random() *
                Math.PI *
                2;

        }


        update(time) {

            this.x += this.vx;

            this.y += this.vy;


            this.phase +=
                0.008;


            // Subtle organic horizontal motion
            this.x +=

                Math.sin(

                    this.phase +
                    time *
                    0.00015

                ) * 0.035;


            // ---------------------------------------------
            // Pointer Interaction
            // ---------------------------------------------

            if (
                pointer.x !== null &&
                pointer.y !== null
            ) {

                const dx =
                    pointer.x -
                    this.x;


                const dy =
                    pointer.y -
                    this.y;


                const distanceSquared =
                    dx * dx +
                    dy * dy;


                // Gently move particles away
                // from the mouse pointer
                if (
                    distanceSquared <
                        28000 &&
                    distanceSquared >
                        1
                ) {

                    const force =
                        0.0024;


                    this.x -=
                        dx *
                        force;


                    this.y -=
                        dy *
                        force;

                }

            }


            // Reset particle when it exits top
            if (
                this.y <
                -30
            ) {

                this.reset();

            }


            // Wrap horizontally
            if (
                this.x <
                -30
            ) {

                this.x =
                    width + 30;

            }


            if (
                this.x >
                width + 30
            ) {

                this.x =
                    -30;

            }

        }


        draw() {

            ctx.beginPath();


            ctx.arc(

                this.x,
                this.y,
                this.radius,
                0,
                Math.PI * 2

            );


            ctx.fillStyle =
                `rgba(
                    208,
                    220,
                    238,
                    ${this.alpha}
                )`;


            ctx.fill();

        }

    }


    // =====================================================
    // 8. RESIZE CANVAS
    // =====================================================

    function resizeCanvas() {

        dpr =
            Math.min(
                window.devicePixelRatio ||
                1,
                2
            );


        width =
            window.innerWidth;


        height =
            window.innerHeight;


        canvas.width =
            Math.floor(
                width *
                dpr
            );


        canvas.height =
            Math.floor(
                height *
                dpr
            );


        canvas.style.width =
            `${width}px`;


        canvas.style.height =
            `${height}px`;


        ctx.setTransform(

            dpr,
            0,
            0,
            dpr,
            0,
            0

        );


        // Dynamic particle count based
        // on screen size
        const particleCount =
            Math.max(

                34,

                Math.min(

                    82,

                    Math.floor(

                        (
                            width *
                            height
                        ) /
                        22000

                    )

                )

            );


        points =
            Array.from(

                {
                    length:
                        particleCount
                },

                () =>
                    new Point()

            );

    }


    // =====================================================
    // 9. DRAW CONNECTIONS BETWEEN PARTICLES
    // =====================================================

    function drawConnections() {

        const maximumDistance =
            Math.min(

                145,

                Math.max(

                    105,

                    width *
                    0.105

                )

            );


        const maximumDistanceSquared =
            maximumDistance *
            maximumDistance;


        for (
            let i = 0;
            i < points.length;
            i++
        ) {

            for (
                let j = i + 1;
                j < points.length;
                j++
            ) {

                const pointA =
                    points[i];


                const pointB =
                    points[j];


                const dx =
                    pointA.x -
                    pointB.x;


                const dy =
                    pointA.y -
                    pointB.y;


                const distanceSquared =
                    dx * dx +
                    dy * dy;


                // Draw line only when
                // particles are close
                if (
                    distanceSquared <
                    maximumDistanceSquared
                ) {

                    const alpha =

                        (
                            1 -
                            distanceSquared /
                            maximumDistanceSquared
                        ) *

                        0.085;


                    ctx.beginPath();


                    ctx.moveTo(

                        pointA.x,
                        pointA.y

                    );


                    ctx.lineTo(

                        pointB.x,
                        pointB.y

                    );


                    ctx.strokeStyle =
                        `rgba(
                            125,
                            164,
                            220,
                            ${alpha}
                        )`;


                    ctx.lineWidth =
                        0.7;


                    ctx.stroke();

                }

            }

        }

    }


    // =====================================================
    // 10. MAIN ANIMATION LOOP
    // =====================================================

    function animate(
        time = 0
    ) {

        ctx.clearRect(

            0,
            0,
            width,
            height

        );


        // Update all particles
        points.forEach(
            (point) => {

                point.update(
                    time
                );

            }
        );


        // Draw neural-network connections
        drawConnections();


        // Draw particles
        points.forEach(
            (point) => {

                point.draw();

            }
        );


        animationFrame =
            window.requestAnimationFrame(
                animate
            );

    }


    // =====================================================
    // 11. POINTER MOVEMENT
    // =====================================================

    function handlePointer(
        event
    ) {

        pointer.x =
            event.clientX;


        pointer.y =
            event.clientY;

    }


    function clearPointer() {

        pointer.x =
            null;


        pointer.y =
            null;

    }


    window.addEventListener(

        "pointermove",

        handlePointer,

        {
            passive: true
        }

    );


    document.addEventListener(

        "pointerleave",

        clearPointer

    );


    // =====================================================
    // 12. HANDLE WINDOW RESIZE
    // =====================================================

    window.addEventListener(

        "resize",

        resizeCanvas

    );


    // =====================================================
    // 13. PAUSE ANIMATION WHEN TAB IS HIDDEN
    // =====================================================

    document.addEventListener(

        "visibilitychange",

        () => {

            if (
                document.hidden &&
                animationFrame
            ) {

                cancelAnimationFrame(
                    animationFrame
                );


                animationFrame =
                    null;

            }

            else if (
                !document.hidden &&
                !animationFrame
            ) {

                animationFrame =
                    requestAnimationFrame(
                        animate
                    );

            }

        }

    );


    // =====================================================
    // 14. START BACKGROUND
    // =====================================================

    resizeCanvas();


    animationFrame =
        requestAnimationFrame(
            animate
        );

})();