import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analyse, generateImage, refineImage, apiError, isImageExpiredError, type RefineHistoryItem } from "../lib/api";
import { Link } from "wouter";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ImagePlus,
  Ruler,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Armchair,
  Lamp,
  Frame,
  Palette,
  Layers,
  X,
  ArrowLeftRight,
  Maximize2,
  Compass,
  RotateCcw,
  CheckCircle2,
  Send,
  Loader2,
  MessageSquare,
  AlertCircle,
} from "lucide-react";


const styles = [
  ["vintage_indian", "Vintage Indian", "Craft, patina, rich history & brass accents"],
  ["contemporary_indian", "Contemporary Indian", "Clean lines, rooted woods & natural stone"],
  ["coastal", "Coastal", "Airy, soft, sun-washed linens & bleached timber"],
  ["industrial", "Industrial", "Raw, graphic, structural steel & masonry"],
  ["scandinavian", "Scandinavian Japandi", "Light, calm, warm minimalism & organic form"],
  ["eclectic", "Eclectic Bohemian", "Collected, expressive, textured & storied"],
  ["shabby_chic_style", "Shabby Chic", "Romantic, lived-in, soft floral & distressed finish"],
  ["mid_century_modern", "Mid-Century Modern", "Iconic silhouettes, walnut & balanced geometry"],
  ["victorian", "Victorian", "Ornate moldings, velvet depth & dramatic poise"],
];

const roomTypes = [
  ["living_room", "Living Room"],
  ["bedroom", "Master Bedroom"],
  ["kitchen", "Culinary Kitchen"],
  ["dining_room", "Dining Room"],
  ["bathroom", "Bath & Spa"],
  ["study_room", "Library & Study"],
];

const STYLE_CATALOG_INFO: Record<string, { prefix: string; suffix: string; wallpaperName: string }> = {
  scandinavian: { prefix: "Minimalist", suffix: "Japandi", wallpaperName: "Japandi Damask Wallpaper" },
  contemporary_indian: { prefix: "Brass-inlay", suffix: "Contemporary Indian", wallpaperName: "Contemporary Indian Textured Linen Wallpaper" },
  vintage_indian: { prefix: "Hand-carved", suffix: "Vintage Indian", wallpaperName: "Vintage Indian Ikat Weave Wallpaper" },
  coastal: { prefix: "Rattan", suffix: "Coastal", wallpaperName: "Coastal Textured Linen Wallpaper" },
  industrial: { prefix: "Iron-frame", suffix: "Industrial", wallpaperName: "Industrial Block Print Wallpaper" },
  eclectic: { prefix: "Woven", suffix: "Bohemian", wallpaperName: "Bohemian Botanical Wallpaper" },
  shabby_chic_style: { prefix: "Floral-print", suffix: "Cottagecore", wallpaperName: "Cottagecore Textured Linen Wallpaper" },
  mid_century_modern: { prefix: "Walnut", suffix: "Mid-Century Modern", wallpaperName: "Mid-Century Modern Geometric Stripe Wallpaper" },
  victorian: { prefix: "Tufted Velvet", suffix: "Victorian", wallpaperName: "Victorian Brocade Wallpaper" },
};

const CATEGORY_COLORS: Record<string, string> = {
  furniture: "#C0603A",
  wallpaper: "#D49B42",
  curtains: "#76876A",
  rugs: "#BCA37F",
  lamps: "#8A9A5B",
  wall_art: "#5B6C7A",
  accent: "#9C6D59",
};

const ROOM_MARKERS = [
  { id: 1, top: "26%", left: "48%", category: "wall_art", itemIndex: 0, label: "Wall Art", alignRight: false },
  { id: 2, top: "38%", left: "80%", category: "lamps", itemIndex: 0, label: "Lighting", alignRight: true },
  { id: 3, top: "62%", left: "28%", category: "furniture", itemIndex: 0, label: "Seating", alignRight: false },
  { id: 4, top: "70%", left: "62%", category: "furniture", itemIndex: 1, label: "Table", alignRight: false },
  { id: 5, top: "84%", left: "46%", category: "rugs", itemIndex: 0, label: "Rug", alignRight: false },
];

export interface RenderedImageBounds {
  width: number;
  height: number;
  top: number;
  left: number;
}

export function parseCoordinate(coord: string | number): number {
  if (typeof coord === "number") {
    return coord <= 1 ? coord : coord / 100;
  }
  const num = parseFloat(coord);
  if (isNaN(num)) return 0;
  return coord.includes("%") || num > 1 ? num / 100 : num;
}

export function calculateImageRenderedBounds(
  containerWidth: number,
  containerHeight: number,
  naturalWidth: number,
  naturalHeight: number
): RenderedImageBounds {
  if (!containerWidth || !containerHeight || !naturalWidth || !naturalHeight) {
    return {
      width: containerWidth || 0,
      height: containerHeight || 0,
      top: 0,
      left: 0,
    };
  }

  const imageAspect = naturalWidth / naturalHeight;
  const containerAspect = containerWidth / containerHeight;

  let width = containerWidth;
  let height = containerHeight;
  let top = 0;
  let left = 0;

  if (imageAspect > containerAspect) {
    // Image is wider than container: horizontal fit, vertical letterbox
    width = containerWidth;
    height = containerWidth / imageAspect;
    top = (containerHeight - height) / 2;
  } else {
    // Image is taller than container: vertical fit, horizontal pillarbox
    height = containerHeight;
    width = containerHeight * imageAspect;
    left = (containerWidth - width) / 2;
  }

  return { width, height, top, left };
}

export function useContainedImageBounds(
  imgRef: React.RefObject<HTMLImageElement | null>,
  src?: string
) {
  const [bounds, setBounds] = useState<RenderedImageBounds | null>(null);

  const updateBounds = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;

    const containerWidth = img.clientWidth || img.parentElement?.clientWidth || 0;
    const containerHeight = img.clientHeight || img.parentElement?.clientHeight || 0;
    const naturalWidth = img.naturalWidth || 0;
    const naturalHeight = img.naturalHeight || 0;

    if (containerWidth && containerHeight && naturalWidth && naturalHeight) {
      setBounds(calculateImageRenderedBounds(containerWidth, containerHeight, naturalWidth, naturalHeight));
    }
  }, [imgRef]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    if (img.complete && img.naturalWidth) {
      updateBounds();
    }

    const ro = new ResizeObserver(() => {
      updateBounds();
    });

    ro.observe(img);
    if (img.parentElement) {
      ro.observe(img.parentElement);
    }

    window.addEventListener("resize", updateBounds);

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", updateBounds);
    };
  }, [imgRef, src, updateBounds]);

  return { bounds, updateBounds };
}

function ItemThumbnailIcon({ category }: { category: string }) {
  const iconProps = { size: 14, style: { color: "#8a7e72" } };
  switch (category) {
    case "furniture":
      return <Armchair {...iconProps} />;
    case "lamps":
    case "lighting":
      return <Lamp {...iconProps} />;
    case "wall_art":
      return <Frame {...iconProps} />;
    case "wallpaper":
      return <Palette {...iconProps} />;
    case "rugs":
    case "rug":
      return <Layers {...iconProps} />;
    default:
      return <Sparkles {...iconProps} />;
  }
}

function getCategoryCatalogItems(
  key: string,
  amount: number,
  selectedStyle: string,
  resultData: Record<string, any> | null
): Array<{ name: string; price: number; thumbnail?: string; category: string }> {
  const explicitItems =
    resultData?.budget_breakdown_items?.[key] ||
    resultData?.budget_items?.[key] ||
    resultData?.category_items?.[key] ||
    resultData?.items?.[key];

  if (Array.isArray(explicitItems) && explicitItems.length > 0) {
    return explicitItems.map((item: any) => ({
      name: item.name || "Catalog item",
      price: Number(item.price || 0),
      thumbnail: item.thumbnail || item.image_url || "",
      category: item.category || key,
    }));
  }

  if (key === "wallpaper") {
    const recoWallpapers = resultData?.recommendations?.wallpapers;
    if (Array.isArray(recoWallpapers) && recoWallpapers.length > 0) {
      return [
        {
          name: recoWallpapers[0].name || "Feature Wallpaper",
          price: amount,
          thumbnail: recoWallpapers[0].image_url || "",
          category: "wallpaper",
        },
      ];
    }
  } else {
    const recoDecor = resultData?.recommendations?.decor_items;
    if (Array.isArray(recoDecor) && recoDecor.length > 0) {
      const matched = recoDecor.filter(
        (d: any) =>
          d.category === key ||
          (key === "curtains" && d.category === "curtain") ||
          (key === "rugs" && d.category === "rug") ||
          (key === "lamps" && d.category === "lighting")
      );
      if (matched.length > 0) {
        return matched.map((m: any) => ({
          name: m.name,
          price: Number(m.price || 0),
          thumbnail: m.image_url || "",
          category: key,
        }));
      }
    }
  }

  const info = STYLE_CATALOG_INFO[selectedStyle] || STYLE_CATALOG_INFO.scandinavian;
  const p = info.prefix;
  const s = info.suffix;

  switch (key) {
    case "wallpaper":
      return [{ name: info.wallpaperName, price: amount, category: "wallpaper" }];
    case "furniture": {
      const seatingPrice = Math.round(amount * 0.3807);
      const tablePrice = Math.round(amount * 0.4338);
      const storagePrice = amount - seatingPrice - tablePrice;
      return [
        { name: `${p} Lounge Seating — ${s}`, price: seatingPrice, category: "furniture" },
        { name: `${p} Sculptural Table — ${s}`, price: tablePrice, category: "furniture" },
        { name: `${p} Low Credenza — ${s}`, price: storagePrice, category: "furniture" },
      ];
    }
    case "curtains":
      return [{ name: `${p} Textured Linen Drapes — ${s}`, price: amount, category: "curtains" }];
    case "rugs":
      return [{ name: `${p} Hand-knotted Wool Rug — ${s}`, price: amount, category: "rugs" }];
    case "lamps":
      return [{ name: `${p} Brass Floor Luminaire — ${s}`, price: amount, category: "lamps" }];
    case "wall_art":
      return [{ name: `${p} Framed Monolith Art — ${s}`, price: amount, category: "wall_art" }];
    case "accent": {
      const mirrorPrice = Math.round(amount * 0.5975);
      const cushionPrice = Math.round(amount * 0.2415);
      const plantPrice = amount - mirrorPrice - cushionPrice;
      return [
        { name: `${p} Arch Wall Mirror — ${s}`, price: mirrorPrice, category: "accent" },
        { name: `${p} Washed Silk Cushions — ${s}`, price: cushionPrice, category: "accent" },
        { name: `${p} Ceramic Planter & Botanical — ${s}`, price: plantPrice, category: "accent" },
      ];
    }
    default:
      return [{ name: `${p} ${key.replace("_", " ")} — ${s}`, price: amount, category: key }];
  }
}

function getAlternativesForItem(
  item: { name: string; price: number; thumbnail?: string; category: string },
  category: string,
  selectedStyle: string,
  resultData: Record<string, any> | null
): Array<{ name: string; price: number; thumbnail?: string; category: string; tier: string }> {
  const extraRecommendations: Array<{ name: string; price: number; thumbnail?: string; category: string }> = [];
  if (category === "wallpaper") {
    const walls = resultData?.recommendations?.wallpapers;
    if (Array.isArray(walls)) {
      walls.forEach((w: any) => {
        if (w.name !== item.name) {
          extraRecommendations.push({
            name: w.name,
            price: Number(w.price || item.price * 1.15),
            thumbnail: w.image_url || "",
            category: "wallpaper",
          });
        }
      });
    }
  } else {
    const decors = resultData?.recommendations?.decor_items;
    if (Array.isArray(decors)) {
      decors.forEach((d: any) => {
        const matchesCat =
          d.category === category ||
          (category === "curtains" && d.category === "curtain") ||
          (category === "rugs" && d.category === "rug") ||
          (category === "lamps" && d.category === "lighting");
        if (matchesCat && d.name !== item.name) {
          extraRecommendations.push({
            name: d.name,
            price: Number(d.price || item.price),
            thumbnail: d.image_url || "",
            category,
          });
        }
      });
    }
  }

  const info = STYLE_CATALOG_INFO[selectedStyle] || STYLE_CATALOG_INFO.scandinavian;
  const p = info.prefix;
  const s = info.suffix;
  const basePrice = item.price || 10000;

  const tierConfigs = [
    {
      tier: "Essential",
      priceMultiplier: 0.75,
      nameModifier: (base: string) => {
        const clean = base.includes("—") ? base.split("—")[0].trim().replace(p, "").trim() : base;
        return `${p} Studio ${clean || "Essential"} — Minimalist Edition`;
      },
    },
    {
      tier: "Atelier",
      priceMultiplier: 1.12,
      nameModifier: (base: string) => {
        const clean = base.includes("—") ? base.split("—")[0].trim().replace(p, "").trim() : base;
        return `${p} Handcrafted ${clean || "Piece"} — ${s}`;
      },
    },
    {
      tier: "Heritage",
      priceMultiplier: 1.48,
      nameModifier: (base: string) => {
        const clean = base.includes("—") ? base.split("—")[0].trim().replace(p, "").trim() : base;
        return `${p} Bespoke ${clean || "Archival"} — Limited Edition`;
      },
    },
  ];

  const results: Array<{ name: string; price: number; thumbnail?: string; category: string; tier: string }> = [];

  if (extraRecommendations.length > 0) {
    extraRecommendations.slice(0, 3).forEach((rec, idx) => {
      results.push({
        name: rec.name,
        price: Math.round(rec.price),
        thumbnail: rec.thumbnail || item.thumbnail,
        category,
        tier: idx === 0 ? "Essential" : idx === 1 ? "Atelier" : "Heritage",
      });
    });
  }

  tierConfigs.forEach((tc) => {
    if (results.length < 3) {
      const altPrice = Math.round(basePrice * tc.priceMultiplier);
      const altName = tc.nameModifier(item.name);
      if (altName !== item.name && !results.some((r) => r.name === altName)) {
        results.push({
          name: altName,
          price: altPrice,
          thumbnail: item.thumbnail,
          category,
          tier: tc.tier,
        });
      }
    }
  });

  return results.slice(0, 3);
}

function DonutChart({
  breakdown,
  total,
  labels,
}: {
  breakdown: Record<string, number>;
  total: number;
  labels: Record<string, string>;
}) {
  const size = 160;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const entries = Object.entries(breakdown).filter(([, amt]) => amt > 0);
  let accumulatedPercent = 0;

  return (
    <div className="donut-chart-container">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ transform: "rotate(-90deg)", borderRadius: "50%" }}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="#ECE7DF"
            strokeWidth={strokeWidth}
          />
          {entries.map(([key, amount]) => {
            const pct = total > 0 ? amount / total : 0;
            const strokeDasharray = `${pct * circumference} ${circumference}`;
            const strokeDashoffset = -accumulatedPercent * circumference;
            accumulatedPercent += pct;
            const color = CATEGORY_COLORS[key] || "#8A9A5B";

            return (
              <circle
                key={key}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="transparent"
                stroke={color}
                strokeWidth={strokeWidth}
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="butt"
                style={{ transition: "stroke-dasharray 0.35s ease, stroke-dashoffset 0.35s ease" }}
              />
            );
          })}
        </svg>
        <div className="donut-center-label">
          <span className="donut-total-tag">Total</span>
          <strong className="donut-total-amt">
            ₹{Math.round(total).toLocaleString("en-IN")}
          </strong>
        </div>
      </div>
      <div className="donut-legend">
        {entries.map(([key, amount]) => {
          const pct = total > 0 ? Math.round((amount / total) * 100) : 0;
          const color = CATEGORY_COLORS[key] || "#8A9A5B";
          return (
            <div key={key} className="donut-legend-item">
              <span className="legend-dot" style={{ background: color }} />
              <span className="legend-name">{labels[key] || key}</span>
              <strong className="legend-pct">{pct}%</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function openFullResolutionImage(src: string) {
  if (!src) return;
  if (src.startsWith("data:")) {
    try {
      const parts = src.split(",");
      const mime = parts[0].match(/:(.*?);/)?.[1] || "image/jpeg";
      const bstr = atob(parts[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      const blob = new Blob([u8arr], { type: mime });
      const blobUrl = URL.createObjectURL(blob);
      window.open(blobUrl, "_blank");
      return;
    } catch {
      const win = window.open("");
      if (win) {
        win.document.write(
          `<!DOCTYPE html><html><head><title>HouseMuse Full Resolution</title><style>html,body{margin:0;padding:0;background:#141210;height:100%;display:flex;align-items:center;justify-content:center;}img{max-width:100%;max-height:100%;object-fit:contain;}</style></head><body><img src="${src}" alt="Full Resolution View" /></body></html>`
        );
        win.document.close();
        return;
      }
    }
  }
  window.open(src, "_blank");
}

export default function Studio() {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [room, setRoom] = useState("living_room");
  const [scope, setScope] = useState("interior_design");
  const [style, setStyle] = useState("contemporary_indian");
  const [budget, setBudget] = useState("180000");
  const [dimensions, setDimensions] = useState({ length: "18", width: "14", height: "10" });
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [resultData, setResultData] = useState<Record<string, any> | null>(null);
  const [afterImage, setAfterImage] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [genError, setGenError] = useState("");
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});
  const [activeItems, setActiveItems] = useState<Record<string, Array<{ name: string; price: number; thumbnail?: string; category: string }>> | null>(null);
  const [activeMarkerId, setActiveMarkerId] = useState<number | null>(null);
  const [activeAltKey, setActiveAltKey] = useState<string | null>(null);

  const afterImgRef = useRef<HTMLImageElement>(null);
  const beforeImgRef = useRef<HTMLImageElement>(null);

  const [projectId, setProjectId] = useState<number | null>(null);
  const [currentImageId, setCurrentImageId] = useState<string>("");
  const [refineHistory, setRefineHistory] = useState<RefineHistoryItem[]>([]);
  const [refineInput, setRefineInput] = useState("");
  const [refining, setRefining] = useState(false);
  const [refineError, setRefineError] = useState("");

  const toggleCategory = (key: string) => {
    setExpandedCategories((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const chosenStyle = useMemo(() => styles.find((s) => s[0] === style)?.[1], [style]);

  const defaultRoomImage = useMemo(() => {
    switch (room) {
      case "bedroom":
        return "/images/bedroom.jpg";
      case "dining_room":
        return "/images/diningroom.jpg";
      case "bathroom":
        return "/images/bathroom.jpg";
      case "kitchen":
        return "/images/kitchen.jpg";
      case "study_room":
        return "/images/lawn.jpg";
      default:
        return "/images/livingroom.jpg";
    }
  }, [room]);

  const resolvedDirectionImage = afterImage || defaultRoomImage;

  const { bounds: afterImageBounds, updateBounds: updateAfterImageBounds } = useContainedImageBounds(
    afterImgRef,
    resolvedDirectionImage
  );

  const { bounds: beforeImageBounds, updateBounds: updateBeforeBounds } = useContainedImageBounds(
    beforeImgRef,
    preview
  );

  const chooseFile = (next: File | undefined) => {
    if (!next || !next.type.startsWith("image/")) return;
    setFile(next);
    setPreview(URL.createObjectURL(next));
  };

  const selectSamplePhoto = async (url: string, name: string) => {
    setPreview(url);
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const f = new File([blob], name, { type: "image/jpeg" });
      setFile(f);
    } catch {
      // preview stays active
    }
  };

  const next = async () => {
    if (step < 4) {
      setStep(step + 1);
      return;
    }
    setLoading(true);
    setErrorMsg("");
    setGenError("");
    try {
      const res = await analyse({
        room_type: room as any,
        length: Number(dimensions.length) * 0.3048,
        width: Number(dimensions.width) * 0.3048,
        height: Number(dimensions.height) * 0.3048,
        styles: [style as any],
        budget: Number(budget),
        scope: scope as any,
        num_rooms: 1,
        harmony: "analogous",
        save: true,
        image: file || undefined,
      });
      setResultData(res?.result || null);
      const currentProjId = res?.project?.id;
      if (currentProjId) {
        setProjectId(currentProjId);
      }
      if (res?.project?.history) {
        setRefineHistory(res.project.history);
      }
      if (res?.project?.generated_image_url) {
        setAfterImage(res.project.generated_image_url);
      }
      if (!preview && res?.project?.source_image) {
        setPreview(res.project.source_image);
      }
      try {
        const imgRes = await generateImage({ project_id: currentProjId, style, room_type: room });
        const b64 = imgRes?.image_b64 || imgRes?.variants?.[0]?.image_b64;
        const fullUrl =
          (b64 ? `data:image/jpeg;base64,${b64}` : "") ||
          imgRes?.image_url ||
          imgRes?.variants?.[0]?.image_url ||
          "";
        if (fullUrl) setAfterImage(fullUrl);
        const genImgId = imgRes?.image_id || imgRes?.variants?.[0]?.image_id || "";
        if (genImgId) setCurrentImageId(genImgId);
      } catch (genErr) {
        // Don't hide failures: show why the AI redesign could not be generated.
        setGenError(apiError(genErr));
      }
      setDone(true);
    } catch (err) {
      setErrorMsg(apiError(err));
    } finally {
      setLoading(false);
    }
  };

  const retry = () => {
    setErrorMsg("");
  };

  const reset = () => {
    setStep(1);
    setDone(false);
    setFile(null);
    setPreview("");
    setResultData(null);
    setAfterImage("");
    setErrorMsg("");
    setExpandedCategories({});
    setActiveItems(null);
    setActiveMarkerId(null);
    setActiveAltKey(null);
    setProjectId(null);
    setCurrentImageId("");
    setRefineHistory([]);
    setRefineInput("");
    setRefining(false);
    setRefineError("");
  };

  const handleRefineSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const prompt = refineInput.trim();
    if (!prompt || refining) return;

    if (!projectId) {
      setRefineError("Please wait for your project to finish saving before refining.");
      return;
    }

    setRefining(true);
    setRefineError("");
    try {
      let res;
      try {
        res = await refineImage({
          project_id: projectId,
          instruction: prompt,
          image_id: currentImageId || undefined,
          current_image_id: currentImageId || undefined,
        });
      } catch (firstErr) {
        // If Flask returns 404 image_expired (e.g. after a restart), resend current image once
        if (isImageExpiredError(firstErr) && afterImage) {
          console.warn("[Studio Refine] image_expired (404), retrying once with current after image...");
          res = await refineImage({
            project_id: projectId,
            instruction: prompt,
            init_image: afterImage,
          });
        } else {
          throw firstErr;
        }
      }

      if (res && res.success) {
        const latestImg =
          (res.image_b64 ? `data:image/jpeg;base64,${res.image_b64}` : "") ||
          res.resulting_image ||
          res.image_url;

        if (latestImg) {
          setAfterImage(latestImg);
        }
        if (res.image_id) {
          setCurrentImageId(res.image_id);
        }

        const newHistoryItem: RefineHistoryItem = {
          id: Date.now(),
          instruction: prompt,
          resulting_image: latestImg,
          created_at: new Date().toISOString(),
          image_id: res.image_id,
        };

        if (res.history && Array.isArray(res.history)) {
          setRefineHistory(res.history);
        } else {
          setRefineHistory((prev) => [...prev, newHistoryItem]);
        }
        setRefineInput("");
      } else {
        // On failure: show error in red notice under chat box, add NO iteration
        setRefineError(res?.error || "Refinement could not be completed. Please try again.");
      }
    } catch (err) {
      // On failure: show error in red notice under chat box, add NO iteration
      setRefineError(apiError(err));
    } finally {
      setRefining(false);
    }
  };


  const BUDGET_LABELS: Record<string, string> = {
    furniture: "Furniture & Seating",
    wallpaper: "Wall Surface & Treatment",
    curtains: "Drapes & Window Textiles",
    rugs: "Rugs & Floor Loom",
    lamps: "Architectural Lighting",
    wall_art: "Curated Wall Art",
    accent: "Material Accents & Flora",
  };

  const baseBreakdown: Record<string, number> = useMemo(() => {
    if (resultData?.budget_breakdown && Object.keys(resultData.budget_breakdown).length > 0) {
      return resultData.budget_breakdown;
    }
    const totalB = Number(budget) || 180000;
    return {
      furniture: Math.round(totalB * 0.4),
      wallpaper: Math.round(totalB * 0.15),
      curtains: Math.round(totalB * 0.12),
      rugs: Math.round(totalB * 0.1),
      lamps: Math.round(totalB * 0.08),
      wall_art: Math.round(totalB * 0.08),
      accent: Math.round(totalB * 0.07),
    };
  }, [resultData, budget]);

  const currentItemsMap = useMemo(() => {
    if (activeItems) return activeItems;
    const initialMap: Record<string, Array<{ name: string; price: number; thumbnail?: string; category: string }>> = {};
    for (const [catKey, catAmt] of Object.entries(baseBreakdown)) {
      initialMap[catKey] = getCategoryCatalogItems(catKey, Number(catAmt), style, resultData);
    }
    return initialMap;
  }, [activeItems, baseBreakdown, style, resultData]);

  const liveCategorySubtotals = useMemo(() => {
    const subtotals: Record<string, number> = {};
    for (const [catKey, itemsList] of Object.entries(currentItemsMap)) {
      subtotals[catKey] = itemsList.reduce((sum, item) => sum + item.price, 0);
    }
    return subtotals;
  }, [currentItemsMap]);

  const liveTotal = useMemo(() => {
    return Object.values(liveCategorySubtotals).reduce((sum, val) => sum + val, 0);
  }, [liveCategorySubtotals]);

  const userBudget = useMemo(() => {
    return Number(budget) || Number(resultData?.budget_total) || 180000;
  }, [budget, resultData]);

  const isOverBudget = useMemo(() => {
    if (resultData?.over_budget) return true;
    return liveTotal > userBudget;
  }, [resultData?.over_budget, liveTotal, userBudget]);

  const shortfallAmount = useMemo(() => {
    if (liveTotal > userBudget) {
      return Math.round(liveTotal - userBudget);
    }
    if (resultData?.shortfall_amount && resultData.shortfall_amount > 0) {
      return Math.round(resultData.shortfall_amount);
    }
    return 0;
  }, [liveTotal, userBudget, resultData?.shortfall_amount]);

  const handleReplaceItem = (
    categoryKey: string,
    itemIdx: number,
    newItem: { name: string; price: number; thumbnail?: string; category: string }
  ) => {
    setActiveItems((prev) => {
      const map = prev ? { ...prev } : { ...currentItemsMap };
      const list = [...(map[categoryKey] || [])];
      list[itemIdx] = newItem;
      return { ...map, [categoryKey]: list };
    });
  };

  if (errorMsg) {
    return (
      <main className="product-page studio-page flex items-center justify-center min-h-[70vh]">
        <div className="editorial-empty-card max-w-lg w-full text-center">
          <div className="empty-glyph text-[#A85832]">
            <Sparkles size={24} />
          </div>
          <span className="eyebrow section-label text-[#A85832]">Spatial Analysis Paused</span>
          <h1 className="headline-serif text-espresso text-3xl my-3">That didn't go through.</h1>
          <p className="text-sm text-[#655E57] mb-6 leading-relaxed">{errorMsg}</p>
          <button className="button primary inline-flex items-center gap-2" onClick={retry}>
            <RotateCcw size={15} /> Try Again
          </button>
        </div>
      </main>
    );
  }

  if (loading) {
    return (
      <main className="product-page studio-page flex items-center justify-center min-h-[75vh]">
        <div className="editorial-loading-box">
          <span className="eyebrow section-label">House Muse / Atelier In Session</span>
          <div className="loading-crosshair-wrap">
            <div className="loading-crosshair-orbit">
              <Compass size={36} className="text-[#8A9A5B]" />
            </div>
          </div>
          <h1 className="headline-serif text-espresso text-3xl md:text-5xl mt-5 mb-3">
            Reading the room.
          </h1>
          <p className="text-sm text-[#655E57] max-w-md mx-auto leading-relaxed mb-6">
            Mapping light axis, proportion, material tactile depth, and catalog allocation into a considered direction.
          </p>
          <div className="editorial-pulse-line">
            <span />
          </div>
          <small className="text-xs text-[#8A7E72] mt-4 block">
            Spatial Analysis in progress · Hand-crafting allocation
          </small>
        </div>
      </main>
    );
  }

  if (done) {
    return (
      <main className="product-page studio-page">
        {/* Results Header */}
        <div className="studio-top">
          <div>
            <span className="eyebrow section-label">Studio / Spatial Direction Ready</span>
            <h1 className="headline-serif text-headline">A room with a point of view.</h1>
            <p className="hero-subhead">
              We interpreted your {room.replaceAll("_", " ")} in a considered {chosenStyle} language.
            </p>
          </div>
          <button className="button ghost" onClick={reset}>
            <RotateCcw size={15} /> Start Another Room
          </button>
        </div>

        {/* Dual Panel Comparison + Breakdown Grid */}
        <div className="result-grid">
          {/* Before Panel with Crosshairs and Detection Markers */}
          <div
            className="image-panel before arch-panel-framed"
            data-image-bounds={beforeImageBounds ? `${Math.round(beforeImageBounds.left)},${Math.round(beforeImageBounds.top)},${Math.round(beforeImageBounds.width)},${Math.round(beforeImageBounds.height)}` : undefined}
          >
            <div className="arch-grid-crosshair top-left" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair top-right" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair bottom-left" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair bottom-right" aria-hidden="true">+</div>

            {preview ? (
              <>
                <div
                  className="image-panel-backdrop"
                  style={{ backgroundImage: `url(${preview})` }}
                  aria-hidden="true"
                />
                <img
                  ref={beforeImgRef}
                  onLoad={updateBeforeBounds}
                  className="image-panel-main"
                  src={preview}
                  alt="Uploaded room photo"
                />
              </>
            ) : (
              <div className="image-placeholder">
                <ImagePlus size={28} />
                <span>Room Source Photo</span>
              </div>
            )}
            <b className="panel-badge-clean">01 / Measured Before</b>
            {preview && (
              <button
                type="button"
                className="floating-expand-btn"
                aria-label="Expand uploaded room"
                onClick={() => openFullResolutionImage(preview)}
              >
                <Maximize2 size={15} />
              </button>
            )}

            {/* Interactive Pins for Detected Objects on Before Photo */}
            {preview &&
              ROOM_MARKERS.map((marker) => {
                const item = currentItemsMap[marker.category]?.[marker.itemIndex] || currentItemsMap[marker.category]?.[0];
                if (!item) return null;
                const isActive = activeMarkerId === marker.id;

                const topPercent = parseCoordinate(marker.top);
                const leftPercent = parseCoordinate(marker.left);
                const markerTop = beforeImageBounds
                  ? `${beforeImageBounds.top + topPercent * beforeImageBounds.height}px`
                  : marker.top;
                const markerLeft = beforeImageBounds
                  ? `${beforeImageBounds.left + leftPercent * beforeImageBounds.width}px`
                  : marker.left;

                return (
                  <div
                    key={marker.id}
                    style={{
                      position: "absolute",
                      top: markerTop,
                      left: markerLeft,
                      transform: "translate(-50%, -50%)",
                      zIndex: isActive ? 45 : 20,
                    }}
                    data-marker-id={marker.id}
                  >
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveMarkerId(isActive ? null : marker.id);
                      }}
                      aria-label={`Marker ${marker.id}: ${item.name}`}
                      className={`marker-pin ${isActive ? "active" : ""}`}
                    >
                      {marker.id}
                    </button>

                    {isActive && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className={`marker-popover ${marker.alignRight ? "align-right" : "align-center"}`}
                      >
                        <div className="popover-top">
                          <span className="popover-cat">
                            {BUDGET_LABELS[marker.category] || marker.category} · #{marker.id}
                          </span>
                          <button
                            type="button"
                            onClick={() => setActiveMarkerId(null)}
                            className="popover-close"
                          >
                            <X size={12} />
                          </button>
                        </div>

                        <div className="popover-body">
                          <div className="popover-thumb">
                            {item.thumbnail ? (
                              <img src={item.thumbnail} alt={item.name} />
                            ) : (
                              <ItemThumbnailIcon category={item.category || marker.category} />
                            )}
                          </div>
                          <div className="popover-meta">
                            <span className="popover-title" title={item.name}>
                              {item.name}
                            </span>
                            <span className="popover-price">
                              ₹{Math.round(item.price).toLocaleString("en-IN")}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>

          {/* After Panel with Crosshairs (Clean generated room direction) */}
          <div
            className="image-panel after arch-panel-framed"
            data-image-bounds={afterImageBounds ? `${Math.round(afterImageBounds.left)},${Math.round(afterImageBounds.top)},${Math.round(afterImageBounds.width)},${Math.round(afterImageBounds.height)}` : undefined}
          >
            <div className="arch-grid-crosshair top-left" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair top-right" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair bottom-left" aria-hidden="true">+</div>
            <div className="arch-grid-crosshair bottom-right" aria-hidden="true">+</div>
            {genError && (
              <div
                role="alert"
                style={{ position: "absolute", left: 12, right: 12, top: 12, zIndex: 20, padding: "10px 12px", borderRadius: 10, background: "rgba(255,255,255,0.95)", color: "#8a2b1a", fontSize: 12, lineHeight: 1.4 }}
              >
                AI redesign could not be generated: {genError}
              </div>
            )}

            <div
              className="image-panel-backdrop"
              style={{ backgroundImage: `url(${resolvedDirectionImage})` }}
              aria-hidden="true"
            />
            <img
              ref={afterImgRef}
              onLoad={updateAfterImageBounds}
              className="image-panel-main"
              src={resolvedDirectionImage}
              alt="Generated interior direction"
            />
            <span className="arch-badge-olive result-badge">
              02 / DIRECTION: {chosenStyle?.toUpperCase()}
            </span>

            <button
              type="button"
              className="floating-expand-btn"
              aria-label="Expand generated room"
              onClick={() => openFullResolutionImage(resolvedDirectionImage)}
            >
              <Maximize2 size={15} />
            </button>
          </div>

          {/* Allocation & Catalog Line-Items */}
          <aside className="breakdown arch-breakdown-card">
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="eyebrow section-label">
                Measured Allocation / ₹{Math.round(liveTotal).toLocaleString("en-IN")}
              </span>
              {isOverBudget && shortfallAmount > 0 && (
                <span className="over-budget-badge">
                  +₹{shortfallAmount.toLocaleString("en-IN")} Over Budget
                </span>
              )}
            </div>
            <h2 className="headline-serif text-espresso text-2xl mb-2">
              Proportion & Budget
            </h2>

            {isOverBudget && shortfallAmount > 0 && (
              <div className="budget-overrun-banner" role="alert">
                <div className="overrun-icon-wrap" aria-hidden="true">
                  <AlertCircle size={15} />
                </div>
                <div className="overrun-content">
                  <strong className="overrun-title">Budget Exceeded</strong>
                  <p className="overrun-message">
                    This direction runs over your ₹{userBudget.toLocaleString("en-IN")} budget by ₹{shortfallAmount.toLocaleString("en-IN")} — try a higher budget or a different style for a closer fit.
                  </p>
                </div>
              </div>
            )}

            {/* Donut Chart */}
            <DonutChart breakdown={liveCategorySubtotals} total={liveTotal} labels={BUDGET_LABELS} />

            {/* Line items with interactive tier alternatives */}
            <div className="breakdown-list">
              {Object.entries(currentItemsMap).map(([key, items]) => {
                const subtotal = liveCategorySubtotals[key] || 0;
                const isExpanded = !!expandedCategories[key];
                const pct = liveTotal > 0 ? (subtotal / liveTotal) * 100 : 0;
                const catColor = CATEGORY_COLORS[key] || "#8A9A5B";

                return (
                  <div className="budget-category-group" key={key}>
                    <div
                      className="budget-row"
                      onClick={() => toggleCategory(key)}
                      role="button"
                      tabIndex={0}
                      aria-expanded={isExpanded}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          toggleCategory(key);
                        }
                      }}
                    >
                      <span className="cat-title-wrap">
                        {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        {BUDGET_LABELS[key] || key}
                        <span className="cat-item-count">
                          {items.length} {items.length === 1 ? "item" : "items"}
                        </span>
                      </span>
                      <strong>₹{Math.round(subtotal).toLocaleString("en-IN")}</strong>
                      <i className="progress-track">
                        <em style={{ width: `${pct}%`, background: catColor }} />
                      </i>
                    </div>

                    {isExpanded && (
                      <div className="alternatives-drawer">
                        <div className="items-list-wrap">
                          {items.map((item, idx) => {
                            const itemKey = `${key}-${idx}`;
                            const isSelectedForAlt = activeAltKey === itemKey;
                            const alternatives = isSelectedForAlt
                              ? getAlternativesForItem(item, key, style, resultData)
                              : [];

                            return (
                              <div key={idx} className="item-slot">
                                <div
                                  onClick={() => setActiveAltKey(isSelectedForAlt ? null : itemKey)}
                                  role="button"
                                  tabIndex={0}
                                  className={`item-slot-header ${isSelectedForAlt ? "selected" : ""}`}
                                >
                                  <div className="item-slot-left">
                                    <div className="item-slot-icon">
                                      {item.thumbnail ? (
                                        <img src={item.thumbnail} alt={item.name} />
                                      ) : (
                                        <ItemThumbnailIcon category={item.category || key} />
                                      )}
                                    </div>
                                    <div className="item-slot-meta">
                                      <span className="item-name" title={item.name}>{item.name}</span>
                                      <span className="item-alt-hint">
                                        <ArrowLeftRight size={9} /> Tap to see alternatives
                                      </span>
                                    </div>
                                  </div>
                                  <div className="item-slot-right">
                                    <span className="item-price">
                                      ₹{Math.round(item.price).toLocaleString("en-IN")}
                                    </span>
                                    <ChevronDown
                                      size={12}
                                      style={{
                                        transform: isSelectedForAlt ? "rotate(180deg)" : "none",
                                        transition: "transform 0.2s ease",
                                      }}
                                    />
                                  </div>
                                </div>

                                {isSelectedForAlt && (
                                  <div className="tier-selector-box">
                                    <div className="tier-header">
                                      <span>Select Price Tier ({alternatives.length} options)</span>
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setActiveAltKey(null);
                                        }}
                                      >
                                        <X size={12} />
                                      </button>
                                    </div>

                                    <div className="tier-options-list">
                                      {alternatives.map((alt, aIdx) => {
                                        const diff = alt.price - item.price;
                                        const isCurrent = alt.name === item.name;
                                        return (
                                          <div
                                            key={aIdx}
                                            onClick={() => {
                                              handleReplaceItem(key, idx, {
                                                name: alt.name,
                                                price: alt.price,
                                                thumbnail: alt.thumbnail || item.thumbnail,
                                                category: key,
                                              });
                                              setActiveAltKey(null);
                                            }}
                                            className={`tier-option-card ${isCurrent ? "active" : ""}`}
                                          >
                                            <div className="tier-card-left">
                                              <span className={`tier-badge tier-${alt.tier.toLowerCase()}`}>
                                                {alt.tier}
                                              </span>
                                              <span className="tier-name" title={alt.name}>{alt.name}</span>
                                            </div>
                                            <div className="tier-card-right">
                                              <div className="tier-price-wrap">
                                                <span className="tier-price">
                                                  ₹{Math.round(alt.price).toLocaleString("en-IN")}
                                                </span>
                                                {diff !== 0 && (
                                                  <span className={`tier-diff ${diff > 0 ? "plus" : "minus"}`}>
                                                    {diff > 0
                                                      ? `+₹${Math.round(diff).toLocaleString("en-IN")}`
                                                      : `-₹${Math.round(Math.abs(diff)).toLocaleString("en-IN")}`}
                                                  </span>
                                                )}
                                              </div>
                                              <button type="button" className={`tier-select-btn ${isCurrent ? "current" : ""}`}>
                                                {isCurrent ? "Current" : "Select"}
                                              </button>
                                            </div>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <p className="fine-print">
              Live spatial allocation: tap any item to explore price-tier alternatives, preview rebalanced totals, and inspect room markers.
            </p>
          </aside>
        </div>

        {/* Studio Image Refinement Chat Panel */}
        <section className="studio-refine-panel" aria-label="Refine Design Direction">
          <div className="refine-panel-header">
            <div className="refine-panel-title-wrap">
              <span className="eyebrow section-label">Iterative Spatial Dialogue</span>
              <h3>Refine Your Direction</h3>
              <p>
                Request targeted spatial adjustments (e.g. &ldquo;make the bed smaller, add a reading chair&rdquo; or &ldquo;warm up the lighting and swap in brass finishes&rdquo;).
              </p>
            </div>
            {refineHistory.length > 0 && (
              <span className="refine-count-pill">
                <MessageSquare size={12} />
                {refineHistory.length} {refineHistory.length === 1 ? "iteration" : "iterations"}
              </span>
            )}
          </div>

          {/* Refinement Thread: User instruction -> Resulting thumbnail */}
          {refineHistory.length > 0 ? (
            <div className="refine-thread">
              {refineHistory.map((item, idx) => (
                <div key={item.id || idx} className="refine-item">
                  {item.resulting_image && (
                    <button
                      type="button"
                      className="refine-thumb-button"
                      title="Click to view full resolution"
                      onClick={() => openFullResolutionImage(item.resulting_image)}
                    >
                      <img src={item.resulting_image} alt={`Iteration #${idx + 1}`} />
                      <div className="refine-thumb-overlay">
                        <Maximize2 size={16} />
                      </div>
                    </button>
                  )}
                  <div className="refine-item-body">
                    <div>
                      <div className="refine-item-top">
                        <span className="refine-item-index">Iteration 0{idx + 1}</span>
                        {item.created_at && (
                          <span className="refine-item-time">
                            {new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        )}
                      </div>
                      <p className="refine-item-text">&ldquo;{item.instruction}&rdquo;</p>
                    </div>
                    {item.resulting_image && (
                      <button
                        type="button"
                        className="refine-apply-btn"
                        onClick={() => setAfterImage(item.resulting_image)}
                      >
                        Set as active direction view →
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="refine-empty-state">
              <div className="refine-empty-icon">
                <Sparkles size={18} />
              </div>
              <p>
                No refinements applied yet. Guide the AI model with conversational cues to evolve furniture scale, materials, palette accents, or layout balance.
              </p>
            </div>
          )}

          {/* Input + Send Button Form */}
          <form onSubmit={handleRefineSubmit} className="refine-input-form">
            <input
              type="text"
              value={refineInput}
              onChange={(e) => setRefineInput(e.target.value)}
              placeholder="e.g. Replace the armchair with a cognac leather lounger, add sheer linen curtains..."
              disabled={refining}
              className="refine-input-field"
            />
            <button
              type="submit"
              disabled={refining || !refineInput.trim()}
              className="button primary refine-send-btn"
            >
              {refining ? (
                <>
                  <Loader2 size={15} className="refine-spin-icon" />
                  <span>Refining...</span>
                </>
              ) : (
                <>
                  <Send size={14} />
                  <span>Send Request</span>
                </>
              )}
            </button>
          </form>

          {/* Loading state indicator */}
          {refining && (
            <div className="refine-loading-banner" role="status" aria-live="polite">
              <Loader2 size={16} className="refine-spin-icon" />
              <span>
                Regenerating spatial composition with img2img conditioning and subtle diffusion passes... This typically takes several seconds.
              </span>
            </div>
          )}

          {/* Error Message */}
          {refineError && (
            <div className="refine-error-banner" role="alert" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <AlertCircle size={16} style={{ flexShrink: 0 }} />
              <span>{refineError}</span>
            </div>
          )}
        </section>
      </main>

    );
  }

  return (
    <main className="product-page studio-page">
      {/* Studio Header */}
      <div className="studio-top">
        <div>
          <span className="eyebrow section-label">House Muse · Atelier & Spatial Studio</span>
          <h1 className="headline-serif text-headline">We Design The Feeling Of Home.</h1>
          <p className="hero-subhead">
            Provide a photograph, spatial coordinates, and an architectural language to shape your room.
          </p>
        </div>
        <Link href="/" className="back-link">
          <ArrowLeft size={15} /> Exit Studio
        </Link>
      </div>

      <div className="studio-layout">
        {/* Horizontal & Vertical Step Track */}
        <aside className="steps-container">
          <div className="steps-track">
            {["Upload", "Room Details", "Style Language", "Budget"].map((label, i) => (
              <button
                key={label}
                type="button"
                className={`step-btn ${step === i + 1 ? "active" : step > i + 1 ? "complete" : ""}`}
                onClick={() => {
                  if (i + 1 < step) setStep(i + 1);
                }}
              >
                <span className="step-circle">
                  {step > i + 1 ? <Check size={13} /> : `0${i + 1}`}
                </span>
                <span className="step-text">{label}</span>
              </button>
            ))}
          </div>

          <div className="studio-aside-note">
            <Ruler size={16} />
            <p>Every allocation begins with the real dimensions of your actual ceiling, walls, and doorway light.</p>
          </div>
        </aside>

        {/* Step Forms */}
        <section className="studio-form-card">
          {/* STEP 1: UPLOAD */}
          {step === 1 && (
            <div className="form-section">
              <span className="eyebrow section-label">01 / The Starting Point</span>
              <h2 className="headline-serif text-espresso text-3xl mb-2">Show us the room as it is.</h2>
              <p className="section-copy">
                One photograph is enough to begin. Natural daylight works best, but lived-in corners are warmly welcomed.
              </p>

              <div
                className="upload-zone arch-panel-framed"
                role="button"
                tabIndex={0}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    inputRef.current?.click();
                  }
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  chooseFile(e.dataTransfer.files[0]);
                }}
              >
                <div className="arch-grid-crosshair top-left" aria-hidden="true">+</div>
                <div className="arch-grid-crosshair top-right" aria-hidden="true">+</div>
                <div className="arch-grid-crosshair bottom-left" aria-hidden="true">+</div>
                <div className="arch-grid-crosshair bottom-right" aria-hidden="true">+</div>

                {preview ? (
                  <div className="preview-container">
                    <img src={preview} alt="Uploaded room preview" />
                    <button
                      type="button"
                      className="floating-expand-btn"
                      aria-label="Expand uploaded room"
                      onClick={(e) => {
                        e.stopPropagation();
                        openFullResolutionImage(preview);
                      }}
                    >
                      <Maximize2 size={15} />
                    </button>
                  </div>
                ) : (
                  <div className="upload-prompt">
                    <div className="upload-icon-circle">
                      <ImagePlus size={26} />
                    </div>
                    <strong>Drop your room photograph here</strong>
                    <span>High resolution JPEG, PNG, or WebP</span>
                    <button
                      type="button"
                      className="browse-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        inputRef.current?.click();
                      }}
                    >
                      Browse Device Photos
                    </button>
                  </div>
                )}
                <input
                  className="native-file-input"
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/*"
                  onChange={(e) => chooseFile(e.target.files?.[0])}
                />
              </div>

              {/* Sample Photo Chooser */}
              <div className="sample-photo-bar">
                <span className="sample-label">Or test with an architectural study:</span>
                <div className="sample-chips">
                  <button
                    type="button"
                    className="sample-chip"
                    onClick={() => selectSamplePhoto("/images/livingroom.jpg", "livingroom_study.jpg")}
                  >
                    Living Room Teak
                  </button>
                  <button
                    type="button"
                    className="sample-chip"
                    onClick={() => selectSamplePhoto("/images/bedroom.jpg", "bedroom_sanctuary.jpg")}
                  >
                    Sanctuary Bedroom
                  </button>
                  <button
                    type="button"
                    className="sample-chip"
                    onClick={() => selectSamplePhoto("/images/diningroom.jpg", "diningroom_gathering.jpg")}
                  >
                    Dining Gathering
                  </button>
                  <button
                    type="button"
                    className="sample-chip"
                    onClick={() => selectSamplePhoto("/images/lawn.jpg", "lawn_veranda_landscape.jpg")}
                  >
                    Veranda Lawn (Landscape)
                  </button>
                </div>
              </div>

              {file && (
                <div className="file-chip">
                  <span className="flex items-center gap-2">
                    <CheckCircle2 size={15} className="text-[#8A9A5B]" /> {file.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setFile(null);
                      setPreview("");
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          )}

          {/* STEP 2: COORDINATES */}
          {step === 2 && (
            <div className="form-section">
              <span className="eyebrow section-label">02 / Spatial Coordinates</span>
              <h2 className="headline-serif text-espresso text-3xl mb-2">Tell us what we’re working with.</h2>
              <p className="section-copy">
                Exact room measurements allow us to allocate furniture scales that breathe naturally in your space.
              </p>

              <div className="dimensions-grid">
                <label className="dimension-field">
                  <span>Length</span>
                  <div className="dim-input-wrap">
                    <input
                      type="number"
                      value={dimensions.length}
                      onChange={(e) => setDimensions({ ...dimensions, length: e.target.value })}
                      min="1"
                    />
                    <small>ft</small>
                  </div>
                </label>
                <label className="dimension-field">
                  <span>Width</span>
                  <div className="dim-input-wrap">
                    <input
                      type="number"
                      value={dimensions.width}
                      onChange={(e) => setDimensions({ ...dimensions, width: e.target.value })}
                      min="1"
                    />
                    <small>ft</small>
                  </div>
                </label>
                <label className="dimension-field">
                  <span>Ceiling Height</span>
                  <div className="dim-input-wrap">
                    <input
                      type="number"
                      value={dimensions.height}
                      onChange={(e) => setDimensions({ ...dimensions, height: e.target.value })}
                      min="1"
                    />
                    <small>ft</small>
                  </div>
                </label>
              </div>

              <div className="select-row-group">
                <label className="field-label">
                  <span>Room Type</span>
                  <select value={room} onChange={(e) => setRoom(e.target.value)}>
                    {roomTypes.map(([id, name]) => (
                      <option value={id} key={id}>{name}</option>
                    ))}
                  </select>
                </label>

                <label className="field-label">
                  <span>Project Scope</span>
                  <select value={scope} onChange={(e) => setScope(e.target.value)}>
                    <option value="home_decor">Home Decor & Accents (Layered curation)</option>
                    <option value="interior_design">Interior Design (Furniture, Lighting & Walls)</option>
                    <option value="full_makeover">Full Architectural Transformation</option>
                  </select>
                </label>
              </div>
            </div>
          )}

          {/* STEP 3: STYLE LANGUAGE */}
          {step === 3 && (
            <div className="form-section">
              <span className="eyebrow section-label">03 / The Atmosphere</span>
              <h2 className="headline-serif text-espresso text-3xl mb-2">Choose a language for the room.</h2>
              <p className="section-copy">
                Pick a foundational direction. You can customize materials and price-tiers once the first allocation is drawn.
              </p>

              <div className="style-grid">
                {styles.map(([id, name, desc]) => (
                  <button
                    type="button"
                    className={`style-option-card ${style === id ? "selected" : ""}`}
                    key={id}
                    onClick={() => setStyle(id)}
                  >
                    <div className="style-option-header">
                      <span className="style-title">{name}</span>
                      {style === id ? (
                        <div className="check-ring active">
                          <Check size={14} />
                        </div>
                      ) : (
                        <div className="check-ring" />
                      )}
                    </div>
                    <small className="style-desc">{desc}</small>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STEP 4: BUDGET BOUNDARY */}
          {step === 4 && (
            <div className="form-section">
              <span className="eyebrow section-label">04 / The Boundary</span>
              <h2 className="headline-serif text-espresso text-3xl mb-2">Give the idea a budget.</h2>
              <p className="section-copy">
                A real number anchors every recommendation to functional catalog items with transparent price tiers.
              </p>

              <div className="budget-hero-display">
                <span className="currency-symbol">₹</span>
                <input
                  value={budget}
                  onChange={(e) => setBudget(e.target.value.replace(/[^0-9]/g, ""))}
                  inputMode="numeric"
                  className="budget-number-input"
                />
                <span className="currency-unit">INR Total Allocation</span>
              </div>

              <div className="range-slider-wrapper">
                <input
                  className="range"
                  type="range"
                  min="50000"
                  max="600000"
                  step="10000"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                />
                <div className="range-labels">
                  <span>₹50,000 (Minimalist Layering)</span>
                  <span>₹6,00,000 (Atelier Full Suite)</span>
                </div>
              </div>

              <div className="save-note">
                <span className="dot" />
                <div>
                  <b>Preserve this direction to your Spatial Portfolio</b>
                  <p>You can revisit the catalog breakdown, trade alternatives, and render new variations anytime.</p>
                </div>
              </div>
            </div>
          )}

          {/* Form Navigation Actions */}
          <div className="form-actions">
            {step > 1 ? (
              <button type="button" className="button ghost" onClick={() => setStep(step - 1)}>
                <ArrowLeft size={15} /> Previous Step
              </button>
            ) : (
              <span />
            )}
            <button type="button" className="button primary" onClick={next}>
              {step === 4 ? "Analyse & Measure Room" : "Continue to Next Step"}
              <ArrowRight size={15} />
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
