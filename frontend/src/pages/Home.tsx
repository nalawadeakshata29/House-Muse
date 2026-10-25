import { useState, useEffect, useRef, useCallback } from "react";
import { Link } from "wouter";
import { ArrowRight, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";

interface Scene {
  id: string;
  name: string;
  kicker: string;
  title: string;
  description: string;
  image: string;
  treatment: string;
  colors: string[];
}

const scenes: Scene[] = [
  {
    id: "living",
    name: "Living Room",
    kicker: "Curated Interiors // Living Sanctuary",
    title: "Poetic Living in Natural Light.",
    description: "Harmonizing warm travertine surfaces, hand-crafted teak timber, and sculptural upholstery designed for contemplative repose.",
    image: "/images/livingroom.jpg",
    treatment: "Minimal Organic Palette",
    colors: ["#F5F2EB", "#8A9A5B", "#3D332B", "#A85832"],
  },
  {
    id: "bedroom",
    name: "Master Suite",
    kicker: "Sanctuary // Bed & Rest",
    title: "Quiet Geometry & Tactile Linen.",
    description: "Soft textural wall treatments, muted olive undertones, and balanced acoustics for deeply restorative evenings.",
    image: "/images/bedroom.jpg",
    treatment: "Japandi Restorative Tone",
    colors: ["#FAF7F2", "#9EAE6F", "#201915", "#D8C7B5"],
  },
  {
    id: "kitchen",
    name: "Culinary Kitchen",
    kicker: "Culinary Craft // Form & Function",
    title: "Architectural Precision Meets Heritage.",
    description: "Monolithic stone counters, brushed unlacquered brass hardware, and bespoke joinery calibrated to your daily ritual.",
    image: "/images/kitchen.jpg",
    treatment: "Contemporary Stone & Brass",
    colors: ["#ECE7DF", "#201915", "#C89D6A", "#75844D"],
  },
  {
    id: "dining",
    name: "Dining Room",
    kicker: "Gathering Space // Intimate Poise",
    title: "Convivial Warmth in Sculptural Form.",
    description: "Centered around a solid fluted walnut table under soft diffuse lighting, curated for lingering dinner conversations.",
    image: "/images/diningroom.jpg",
    treatment: "Warm Walnut & Terracotta",
    colors: ["#F5F2EB", "#A85832", "#3D332B", "#E4DCD0"],
  },
  {
    id: "bath",
    name: "Bath & Spa",
    kicker: "Sensory Bath // Water & Stone",
    title: "Rituals Carved in Raw Materiality.",
    description: "Microcement finishes, seamless floor-to-ceiling glass, and matte black brassware for a private thermal retreat.",
    image: "/images/bathroom.jpg",
    treatment: "Microcement & Matte Finish",
    colors: ["#ECE7DF", "#7E746A", "#201915", "#8A9A5B"],
  },
];

const SLIDE_DURATION = 5000;

export default function Home() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [mousePos, setMousePos] = useState({ x: -500, y: -500 });
  const [isMoving, setIsMoving] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const wheelLock = useRef(false);
  const touchStartY = useRef<number | null>(null);
  const touchStartX = useRef<number | null>(null);

  const goToSlide = useCallback((nextIndex: number) => {
    setActiveIndex((nextIndex + scenes.length) % scenes.length);
  }, []);

  const nextSlide = useCallback(() => {
    goToSlide(activeIndex + 1);
  }, [activeIndex, goToSlide]);

  const prevSlide = useCallback(() => {
    goToSlide(activeIndex - 1);
  }, [activeIndex, goToSlide]);

  // Slideshow automatic progression timer
  useEffect(() => {
    if (isPaused) return;
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % scenes.length);
    }, SLIDE_DURATION);

    return () => clearInterval(interval);
  }, [isPaused, activeIndex]);

  // Mouse move flashlight cursor effect
  useEffect(() => {
    let timeout: NodeJS.Timeout;
    const handleMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
      setIsMoving(true);
      clearTimeout(timeout);
      timeout = setTimeout(() => setIsMoving(false), 800);
    };

    window.addEventListener("mousemove", handleMove);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      clearTimeout(timeout);
    };
  }, []);

  // Wheel and keyboard navigation (slideshow pattern)
  useEffect(() => {
    const onWheel = (event: WheelEvent) => {
      if (wheelLock.current || Math.abs(event.deltaY) < 15) return;
      wheelLock.current = true;
      if (event.deltaY > 0) {
        nextSlide();
      } else {
        prevSlide();
      }
      window.setTimeout(() => {
        wheelLock.current = false;
      }, 700);
    };

    const onKey = (event: KeyboardEvent) => {
      if (["ArrowRight", "ArrowDown", " "].includes(event.key)) {
        event.preventDefault();
        nextSlide();
      } else if (["ArrowLeft", "ArrowUp"].includes(event.key)) {
        event.preventDefault();
        prevSlide();
      }
    };

    window.addEventListener("wheel", onWheel, { passive: true });
    window.addEventListener("keydown", onKey);

    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("keydown", onKey);
    };
  }, [nextSlide, prevSlide]);

  // Touch swipe support for mobile
  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartY.current = e.touches[0].clientY;
    touchStartX.current = e.touches[0].clientX;
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (touchStartY.current === null || touchStartX.current === null) return;
    const diffY = touchStartY.current - e.changedTouches[0].clientY;
    const diffX = touchStartX.current - e.changedTouches[0].clientX;

    if (Math.abs(diffX) > Math.abs(diffY)) {
      if (Math.abs(diffX) > 40) {
        if (diffX > 0) nextSlide();
        else prevSlide();
      }
    } else {
      if (Math.abs(diffY) > 40) {
        if (diffY > 0) nextSlide();
        else prevSlide();
      }
    }

    touchStartY.current = null;
    touchStartX.current = null;
  };

  const current = scenes[activeIndex];

  return (
    <div
      className="house-hero"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      style={
        {
          "--font-display": "var(--font-display, 'Fraunces', Georgia, serif)",
          fontFamily: "var(--font-body)",
        } as React.CSSProperties
      }
    >
      {/* Background Image Slideshow with Smooth Crossfade */}
      <div className="house-scenes" aria-label="Interior scenes slideshow">
        {scenes.map((scene, idx) => (
          <div
            key={scene.id}
            className={`house-scene ${idx === activeIndex ? "active" : ""}`}
            style={{ backgroundImage: `url("${scene.image}")` }}
            role="img"
            aria-label={scene.name}
          />
        ))}
      </div>

      <div className="house-vignette" />

      <div
        className={`cursor-light ${isMoving ? "moving" : ""}`}
        style={
          {
            "--pointer-x": `${mousePos.x}px`,
            "--pointer-y": `${mousePos.y}px`,
          } as React.CSSProperties
        }
      />

      <div className="house-annotation top-left">
        <i />
        <span style={{ fontFamily: "var(--font-body)", color: "rgba(250, 247, 242, 0.8)" }}>Architectural Intelligence // V2.0</span>
      </div>

      <div className="house-annotation top-right">
        <span style={{ fontFamily: "var(--font-body)", color: "rgba(250, 247, 242, 0.8)" }}>Harmonic Space Synthesis</span>
        <i />
      </div>

      {/* Hero Copy (Pauses auto-advance on hover so user can read / click) */}
      <div
        className="house-copy"
        onMouseEnter={() => setIsPaused(true)}
        onMouseLeave={() => setIsPaused(false)}
      >
        <div key={current.id} className="scene-copy-transition">
          <span className="scene-kicker eyebrow" style={{ color: "var(--olive-light)" }}>{current.kicker}</span>
          <h1
            className="headline-serif"
            style={{
              fontFamily: "var(--font-display)",
              color: "var(--cream-light, #FAF7F2)",
              fontVariationSettings: "var(--font-fraunces-soft)",
            }}
          >
            {current.title}
          </h1>
          <p className="hero-subhead" style={{ color: "rgba(245, 242, 235, 0.85)" }}>{current.description}</p>
        </div>
        <Link href="/studio" className="door-cta" style={{ fontFamily: "var(--font-body)" }}>
          <span className="door-icon">
            <ArrowRight size={15} />
          </span>
          <span>Step into the Studio</span>
        </Link>
      </div>

      {/* Room Treatment / Palette details */}
      <div className="room-treatment">
        <div key={current.id} className="scene-copy-transition" style={{ display: "flex", alignItems: "flex-end", gap: "14px" }}>
          <div className="treatment-label" style={{ fontFamily: "var(--font-body)", color: "rgba(250, 247, 242, 0.8)" }}>{current.treatment}</div>
          <div className="treatment-rule" />
          <div className="treatment-palette">
            {current.colors.map((c, i) => (
              <span key={i} style={{ backgroundColor: c }} />
            ))}
          </div>
        </div>
      </div>

      <div className="house-metrics">
        <div>
          <b style={{ fontFamily: "var(--font-display)", color: "var(--cream-light, #FAF7F2)" }}>98.4%</b>
          <span style={{ fontFamily: "var(--font-body)", color: "rgba(245, 242, 235, 0.7)" }}>Aesthetic Harmony</span>
        </div>
        <div>
          <b style={{ fontFamily: "var(--font-display)", color: "var(--cream-light, #FAF7F2)" }}>&lt; 3.2s</b>
          <span style={{ fontFamily: "var(--font-body)", color: "rgba(245, 242, 235, 0.7)" }}>Neural Render</span>
        </div>
        <div>
          <b style={{ fontFamily: "var(--font-display)", color: "var(--cream-light, #FAF7F2)" }}>9 Styles</b>
          <span style={{ fontFamily: "var(--font-body)", color: "rgba(245, 242, 235, 0.7)" }}>Curated Palettes</span>
        </div>
      </div>

      {/* Slide Counter / Cue */}
      <div className="scroll-cue" style={{ fontFamily: "var(--font-body)" }}>
        <Sparkles size={13} style={{ color: "var(--olive-light)" }} />
        <span>Scene 0{activeIndex + 1} / 0{scenes.length}</span>
      </div>

      {/* Interactive Scene Navigation with Slideshow Prev/Next arrows */}
      <div
        className="scene-nav"
        role="tablist"
        aria-label="Room scenes slideshow"
        style={{ fontFamily: "var(--font-body)" }}
        onMouseEnter={() => setIsPaused(true)}
        onMouseLeave={() => setIsPaused(false)}
      >
        <button
          type="button"
          className="scene-nav-arrow"
          onClick={prevSlide}
          aria-label="Previous scene"
          title="Previous slide"
        >
          <ChevronLeft size={14} />
        </button>

        {scenes.map((scene, idx) => (
          <button
            key={scene.id}
            role="tab"
            aria-selected={idx === activeIndex}
            className={idx === activeIndex ? "active" : ""}
            onClick={() => goToSlide(idx)}
            style={{ fontFamily: "var(--font-body)" }}
          >
            <span>0{idx + 1}</span>
            {scene.name}
          </button>
        ))}

        <button
          type="button"
          className="scene-nav-arrow"
          onClick={nextSlide}
          aria-label="Next scene"
          title="Next slide"
        >
          <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}
