/**
 * Career Centerpiece 3D Engine & Interactive Waypoint Driver
 * 
 * Drives the 3D spatial perspective of the career timeline across:
 * 1. Kotak Mahindra Bank (Commercial Banking AI & Risk Systems)
 * 2. STMicroelectronics (Embedded Systems, Firmware & IEEE Publication)
 * 3. BITS Pilani (Academic Dual Degree, Quant Competition Global #4)
 * 
 * Features:
 * - Scroll-linked 3D stage rotation and depth translation (Z-axis)
 * - Interactive waypoint scrubber buttons with cubic-bezier easing
 * - Canvas 2D orbital particle rail connecting career waypoints
 * - Graceful fallback respecting `prefers-reduced-motion: reduce`
 */

(function () {
  const container = document.getElementById("career-centerpiece");
  const stage = document.getElementById("career-stage");
  const nodes = document.querySelectorAll(".waypoint-node");
  const scrubBtns = document.querySelectorAll(".scrub-btn");
  const canvas = document.getElementById("career-canvas");

  if (!container || !stage || nodes.length === 0) return;

  const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  let isReducedMotion = reducedMotionQuery.matches;

  reducedMotionQuery.addEventListener("change", (e) => {
    isReducedMotion = e.matches;
    resetTransforms();
  });

  let activeIndex = 0;
  let targetProgress = 0;
  let currentProgress = 0;
  let isScrubbing = false;

  // Waypoint angles and Z positions in 3D stage
  const WAYPOINTS = [
    { rotY: 0, rotX: 4, z: 20 },      // 0: Kotak (Foreground Front)
    { rotY: -16, rotX: 8, z: -80 },   // 1: STMicro (Mid Depth, tilted)
    { rotY: 18, rotX: 12, z: -180 }   // 2: BITS Pilani (Deep Origin)
  ];

  function setWaypoint(idx, smooth = true) {
    activeIndex = Math.max(0, Math.min(WAYPOINTS.length - 1, idx));
    targetProgress = activeIndex / (WAYPOINTS.length - 1);

    // Update scrubber buttons
    scrubBtns.forEach((btn, i) => {
      btn.classList.toggle("active", i === activeIndex);
      btn.setAttribute("aria-selected", i === activeIndex ? "true" : "false");
    });

    // Update node states
    nodes.forEach((node, i) => {
      const isActive = i === activeIndex;
      node.classList.toggle("active-focus", isActive);
      if (isReducedMotion) {
        node.style.transform = "none";
        node.style.opacity = "1";
      } else {
        const offset = i - activeIndex;
        const zDist = -Math.abs(offset) * 120;
        const yRot = offset * 14;
        const opacity = isActive ? 1 : Math.max(0.4, 1 - Math.abs(offset) * 0.4);
        node.style.transform = `translate3d(0, ${offset * 20}px, ${zDist}px) rotateY(${yRot}deg)`;
        node.style.opacity = opacity.toString();
      }
    });

    if (smooth && !isReducedMotion) {
      const wp = WAYPOINTS[activeIndex];
      stage.style.transform = `rotateX(${wp.rotX}deg) rotateY(${wp.rotY}deg) translateZ(${wp.z}px)`;
    }
  }

  function resetTransforms() {
    if (isReducedMotion) {
      stage.style.transform = "none";
      nodes.forEach(n => {
        n.style.transform = "none";
        n.style.opacity = "1";
      });
    } else {
      setWaypoint(activeIndex, false);
    }
  }

  // Scrubber click listeners
  scrubBtns.forEach((btn, i) => {
    btn.addEventListener("click", () => {
      isScrubbing = true;
      setWaypoint(i, true);
      setTimeout(() => { isScrubbing = false; }, 600);
    });
  });

  // Scroll driver
  let ticking = false;
  function onScroll() {
    if (isScrubbing || isReducedMotion) return;
    if (!ticking) {
      requestAnimationFrame(() => {
        const rect = container.getBoundingClientRect();
        const winH = window.innerHeight;
        
        // Calculate relative position within viewport
        const start = winH * 0.7;
        const end = -rect.height * 0.5;
        const progress = Math.max(0, Math.min(1, (start - rect.top) / (start - end)));

        // Determine which milestone to focus
        let targetIdx = 0;
        if (progress > 0.66) targetIdx = 2;
        else if (progress > 0.33) targetIdx = 1;
        else targetIdx = 0;

        if (targetIdx !== activeIndex) {
          setWaypoint(targetIdx, true);
        }
        ticking = false;
      });
      ticking = true;
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });

  // Canvas 2D Orbital Connector Rail
  if (canvas) {
    const ctx = canvas.getContext("2d");
    let particles = [];
    const PARTICLE_COUNT = 36;

    function resizeCanvas() {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    }
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 2 + 1,
        alpha: Math.random() * 0.4 + 0.2
      });
    }

    function renderRail() {
      if (isReducedMotion) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw subtle orbital trajectory lines connecting cards
      ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(canvas.width * 0.2, 50);
      ctx.bezierCurveTo(
        canvas.width * 0.8, canvas.height * 0.35,
        canvas.width * 0.1, canvas.height * 0.65,
        canvas.width * 0.5, canvas.height - 40
      );
      ctx.stroke();

      // Render flowing energy particles
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.fillStyle = `rgba(56, 189, 248, ${p.alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      requestAnimationFrame(renderRail);
    }

    if (!isReducedMotion) {
      requestAnimationFrame(renderRail);
    }
  }

  // Initialize
  setWaypoint(0, false);
})();
