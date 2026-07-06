import React, { useEffect, useRef, useState, useCallback } from "react";

/**
 * THIDA 姿勢歪みチェック
 * - 正面カメラで骨格検出（TensorFlow.js MoveNet, CDN経由）
 * - 施術前 / 施術後を撮影して歪みスコアを比較
 * - お客様ごとに履歴を保存
 *   - Claudeアーティファクト環境では window.storage を使用
 *   - 通常のWeb環境（Vercel等）にデプロイした場合は localStorage にフォールバック
 *     （storageAdapter が自動判定）
 */

const TFJS_SCRIPTS = [
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.20.0/dist/tf-core.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.20.0/dist/tf-converter.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.20.0/dist/tf-backend-webgl.min.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js",
];

function loadScript(src, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.crossOrigin = "anonymous";
    const timer = setTimeout(() => {
      reject(new Error(`timeout loading ${src}`));
    }, timeoutMs);
    s.onload = () => {
      clearTimeout(timer);
      resolve();
    };
    s.onerror = () => {
      clearTimeout(timer);
      reject(new Error(`failed to load ${src}`));
    };
    document.body.appendChild(s);
  });
}

// キーポイント名（MoveNet 17点）
const KP = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

const SKELETON_EDGES = [
  ["leftShoulder", "rightShoulder"],
  ["leftHip", "rightHip"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
  ["leftEar", "leftShoulder"],
  ["rightEar", "rightShoulder"],
];

function angleFromHorizontal(a, b) {
  // 水平線からの傾き（度数）。正=右下がり
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * 歪みスコア計算（0=歪みなし理想, 100=重度）
 * 正面1枚の情報から算出できる指標に限定:
 *  - 肩の水平差（左右の高さの差）
 *  - 骨盤（腰）の水平差
 *  - 首の傾き（耳の中点と肩の中点のズレ）
 *  - 体幹の左右シフト（肩の中点と骨盤の中点の水平ズレ）
 */
function computeDistortion(keypoints) {
  const byName = {};
  keypoints.forEach((k, i) => {
    const name = Object.keys(KP).find((n) => KP[n] === i);
    if (name) byName[name] = k;
  });

  const minScore = 0.3;
  const required = [
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftEar",
    "rightEar",
  ];
  const missing = required.filter(
    (n) => !byName[n] || (byName[n].score ?? 1) < minScore
  );
  if (missing.length > 0) {
    return { ok: false, missing };
  }

  const ls = byName.leftShoulder;
  const rs = byName.rightShoulder;
  const lh = byName.leftHip;
  const rh = byName.rightHip;
  const le = byName.leftEar;
  const re = byName.rightEar;

  // 体格差による正規化のため肩幅を基準スケールにする
  const shoulderWidth = dist(ls, rs) || 1;

  const shoulderTilt = Math.abs(angleFromHorizontal(ls, rs)); // 度
  const hipTilt = Math.abs(angleFromHorizontal(lh, rh)); // 度

  const shoulderMid = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };
  const hipMid = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
  const earMid = { x: (le.x + re.x) / 2, y: (le.y + re.y) / 2 };

  // 体幹の左右シフト（肩中点と骨盤中点の水平ズレ / 肩幅で正規化）
  const trunkShiftRatio = Math.abs(shoulderMid.x - hipMid.x) / shoulderWidth;

  // 首の傾き（耳中点と肩中点の水平ズレ / 肩幅で正規化）
  const neckShiftRatio = Math.abs(earMid.x - shoulderMid.x) / shoulderWidth;

  // それぞれを0-100の部分点に変換（経験則ベースの閾値）
  const shoulderScore = Math.min(100, (shoulderTilt / 12) * 100); // 12度でほぼ満点
  const hipScore = Math.min(100, (hipTilt / 10) * 100); // 10度でほぼ満点
  const trunkScore = Math.min(100, (trunkShiftRatio / 0.18) * 100);
  const neckScore = Math.min(100, (neckShiftRatio / 0.15) * 100);

  // 総合スコア（重み付け平均） 0=良好 100=歪み大
  const total =
    shoulderScore * 0.3 + hipScore * 0.3 + trunkScore * 0.25 + neckScore * 0.15;

  return {
    ok: true,
    total: Math.round(total),
    details: {
      shoulderTilt: Math.round(shoulderTilt * 10) / 10,
      hipTilt: Math.round(hipTilt * 10) / 10,
      trunkShiftRatio: Math.round(trunkShiftRatio * 1000) / 1000,
      neckShiftRatio: Math.round(neckShiftRatio * 1000) / 1000,
      shoulderScore: Math.round(shoulderScore),
      hipScore: Math.round(hipScore),
      trunkScore: Math.round(trunkScore),
      neckScore: Math.round(neckScore),
      shoulderHigherSide: ls.y < rs.y ? "left" : "right", // yが小さい=高い
      hipHigherSide: lh.y < rh.y ? "left" : "right",
    },
    keypointsSnapshot: byName,
  };
}

function scoreLabel(score) {
  if (score <= 20) return { label: "良好", color: "#3fae6a" };
  if (score <= 40) return { label: "軽度の歪み", color: "#e0b23a" };
  if (score <= 65) return { label: "歪みあり", color: "#e07a3a" };
  return { label: "強い歪み", color: "#d1473f" };
}

// スコアと部位別詳細から、おすすめトレーニング/ストレッチを提案
function recommendTraining(result) {
  if (!result || !result.ok) return [];
  const d = result.details;
  const recs = [];

  if (d.shoulderScore >= 25) {
    const side = d.shoulderHigherSide === "left" ? "右" : "左";
    recs.push({
      title: "肩甲骨まわりのストレッチ",
      detail: `肩の高さに左右差があります。${side}肩を下げる意識で、肩甲骨はがし・タオルを使った肩回しを1日2セット行いましょう。`,
    });
  }
  if (d.hipScore >= 25) {
    const side = d.hipHigherSide === "left" ? "右" : "左";
    recs.push({
      title: "骨盤まわりの筋膜リリース＋股関節ストレッチ",
      detail: `骨盤の高さに左右差があります。${side}側の中臀筋・腸腰筋を中心にほぐし、ヒップリフトで骨盤の安定性を高めましょう。`,
    });
  }
  if (d.trunkScore >= 25) {
    recs.push({
      title: "体幹（インナーマッスル）加圧トレーニング",
      detail: "上半身と骨盤の中心軸がズレています。加圧トレーニングでの体幹プランク・サイドプランクが効果的です。",
    });
  }
  if (d.neckScore >= 25) {
    recs.push({
      title: "首・僧帽筋のリラックスストレッチ",
      detail: "頭の位置が体の中心からズレています。あご引きエクササイズとストレートネック予防ストレッチを取り入れましょう。",
    });
  }
  if (recs.length === 0) {
    recs.push({
      title: "現状維持コンディショニング",
      detail: "大きな歪みは見られません。全身の軽めの加圧トレーニングで今の状態をキープしましょう。",
    });
  }
  return recs;
}

// 歪みスコアから次回整体のおすすめ日数（シンプルなルール）
function recommendNextVisit(score) {
  if (score <= 20) return { days: "14〜21日後", note: "良好な状態です。予防メンテナンス目的で2〜3週間後がおすすめです。" };
  if (score <= 40) return { days: "10〜14日後", note: "軽い歪みが見られます。定着を防ぐため10日〜2週間後の来店がおすすめです。" };
  if (score <= 65) return { days: "5〜7日後", note: "歪みが出ています。癖がつく前に1週間程度での再施術をおすすめします。" };
  return { days: "2〜4日後", note: "強い歪みが見られます。短いスパンでの集中的なケアをおすすめします。" };
}

function useTfPoseDetector() {
  const [status, setStatus] = useState("idle"); // idle|loading|ready|error
  const [errorDetail, setErrorDetail] = useState(null);
  const [progress, setProgress] = useState("");
  const progressRef = useRef("");
  const detectorRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      setStatus("loading");
      try {
        for (const src of TFJS_SCRIPTS) {
          progressRef.current = src.split("/").pop();
          setProgress(progressRef.current);
          await loadScript(src);
        }
        progressRef.current = "tf.setBackend";
        setProgress(progressRef.current);
        // eslint-disable-next-line no-undef
        await tf.setBackend("webgl");
        progressRef.current = "tf.ready";
        setProgress(progressRef.current);
        // eslint-disable-next-line no-undef
        await tf.ready();
        progressRef.current = "createDetector";
        setProgress(progressRef.current);
        // eslint-disable-next-line no-undef
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );
        if (cancelled) return;
        detectorRef.current = detector;
        setStatus("ready");
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setErrorDetail((e && e.message ? e.message : String(e)) + " / at: " + progressRef.current);
          setStatus("error");
        }
      }
    }
    init();
    return () => {
      cancelled = true;
    };
  }, []);

  return { status, detectorRef, errorDetail, progress };
}

function CameraCapture({ onCaptured, disabled }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState(null);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 720, height: 1280 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch (e) {
      console.error(e);
      setError("カメラを起動できませんでした。ブラウザのカメラ権限をご確認ください。");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraOn(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const capture = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    onCaptured(canvas);
  };

  return (
    <div className="camera-box">
      <div className="camera-frame">
        <video ref={videoRef} playsInline muted className="camera-video" />
        {!cameraOn && (
          <div className="camera-placeholder">
            <span>カメラ映像がここに表示されます</span>
          </div>
        )}
      </div>
      {error && <p className="error-text">{error}</p>}
      <div className="camera-controls">
        {!cameraOn ? (
          <button className="btn btn-primary" onClick={startCamera} disabled={disabled}>
            カメラを起動
          </button>
        ) : (
          <>
            <button className="btn btn-primary" onClick={capture} disabled={disabled}>
              正面から撮影する
            </button>
            <button className="btn btn-ghost" onClick={stopCamera}>
              カメラを停止
            </button>
          </>
        )}
      </div>
      <p className="hint-text">
        全身が縦に収まるよう、2〜3m離れて正面を向いて立ってください。
      </p>
    </div>
  );
}

function ResultOverlayCanvas({ imageEl, result, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageEl) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(imageEl, 0, 0, width, height);

    if (result && result.ok) {
      const kp = result.keypointsSnapshot;
      ctx.strokeStyle = "#5aa9c7";
      ctx.lineWidth = 3;
      SKELETON_EDGES.forEach(([a, b]) => {
        const pa = kp[a];
        const pb = kp[b];
        if (!pa || !pb) return;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      });
      ctx.fillStyle = "#f2c14e";
      Object.values(kp).forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });

      // 水平基準線（肩・骨盤）
      const ls = kp.leftShoulder,
        rs = kp.rightShoulder,
        lh = kp.leftHip,
        rh = kp.rightHip;
      if (ls && rs) {
        ctx.strokeStyle = "rgba(232,90,60,0.85)";
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(ls.x, ls.y);
        ctx.lineTo(rs.x, rs.y);
        ctx.stroke();
      }
      if (lh && rh) {
        ctx.strokeStyle = "rgba(58,127,224,0.85)";
        ctx.beginPath();
        ctx.moveTo(lh.x, lh.y);
        ctx.lineTo(rh.x, rh.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }, [imageEl, result, width, height]);

  return <canvas ref={canvasRef} className="overlay-canvas" />;
}

function ScoreGauge({ score }) {
  const { label, color } = scoreLabel(score);
  const circumference = 2 * Math.PI * 54;
  const offset = circumference * (1 - Math.min(score, 100) / 100);
  return (
    <div className="gauge-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="54" fill="none" stroke="#22303f" strokeWidth="12" />
        <circle
          cx="70"
          cy="70"
          r="54"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        <text x="70" y="66" textAnchor="middle" fontSize="30" fontWeight="800" fill="#f4efe6">
          {score}
        </text>
        <text x="70" y="88" textAnchor="middle" fontSize="12" fill="#9aa7b5">
          歪みスコア
        </text>
      </svg>
      <div className="gauge-label" style={{ color }}>
        {label}
      </div>
    </div>
  );
}

function ResultDetails({ result }) {
  if (!result || !result.ok) return null;
  const d = result.details;
  return (
    <div className="detail-grid">
      <div className="detail-item">
        <span className="detail-name">肩の傾き</span>
        <span className="detail-value">{d.shoulderTilt}°</span>
      </div>
      <div className="detail-item">
        <span className="detail-name">骨盤の傾き</span>
        <span className="detail-value">{d.hipTilt}°</span>
      </div>
      <div className="detail-item">
        <span className="detail-name">体幹の左右シフト</span>
        <span className="detail-value">{(d.trunkShiftRatio * 100).toFixed(1)}%</span>
      </div>
      <div className="detail-item">
        <span className="detail-name">首の傾き</span>
        <span className="detail-value">{(d.neckShiftRatio * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

// ---- お客様管理 ----

function genId() {
  return "c_" + Math.random().toString(36).slice(2, 10);
}

// storageのvalueが文字列(JSON)でもオブジェクトでもどちらでも読めるようにする
function parseStoredValue(value) {
  if (value == null) return null;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch (e) {
      return value;
    }
  }
  return value;
}

/**
 * window.storage（Claudeアーティファクト環境専用API）が使えない場合、
 * localStorage を使った互換アダプタにフォールバックする。
 * 自分のWebホスティング環境（Vercel等）にデプロイした際もこの分岐で動作する。
 */
const storageAdapter = {
  hasNative: () => typeof window !== "undefined" && window.storage && typeof window.storage.set === "function",

  async set(key, value) {
    if (this.hasNative()) {
      return window.storage.set(key, value);
    }
    try {
      localStorage.setItem("thida:" + key, JSON.stringify(value));
      return { key, value, shared: false };
    } catch (e) {
      console.error("localStorage set failed", e);
      return null;
    }
  },

  async get(key) {
    if (this.hasNative()) {
      return window.storage.get(key);
    }
    try {
      const raw = localStorage.getItem("thida:" + key);
      if (raw == null) return null;
      return { key, value: JSON.parse(raw), shared: false };
    } catch (e) {
      return null;
    }
  },

  async list(prefix) {
    if (this.hasNative()) {
      return window.storage.list(prefix);
    }
    try {
      const keys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith("thida:" + (prefix || ""))) {
          keys.push(k.slice("thida:".length));
        }
      }
      return { keys };
    } catch (e) {
      return { keys: [] };
    }
  },
};

function CustomerPicker({ customers, currentId, onSelect, onCreate }) {
  const [newName, setNewName] = useState("");
  return (
    <div className="customer-picker">
      <label className="field-label">お客様を選択</label>
      <select
        className="select-input"
        value={currentId || ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="">-- 選択してください --</option>
        {customers.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
          </option>
        ))}
      </select>
      <div className="customer-new">
        <input
          className="text-input"
          placeholder="新規お客様の名前"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
        />
        <button
          className="btn btn-secondary"
          onClick={() => {
            if (!newName.trim()) return;
            const id = genId();
            onCreate({ id, name: newName.trim(), createdAt: Date.now() });
            setNewName("");
          }}
        >
          追加
        </button>
      </div>
    </div>
  );
}

export default function PostureCheckApp() {
  const { status: modelStatus, detectorRef, errorDetail: modelErrorDetail, progress: modelProgress } = useTfPoseDetector();
  const [phase, setPhase] = useState("before"); // before | after | summary
  const [beforeShot, setBeforeShot] = useState(null); // {imageEl, result}
  const [afterShot, setAfterShot] = useState(null);
  const [busy, setBusy] = useState(false);
  const [customers, setCustomers] = useState([]);
  const [currentCustomerId, setCurrentCustomerId] = useState("");
  const [history, setHistory] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [saving, setSaving] = useState(false);

  // 顧客一覧をロード
  useEffect(() => {
    (async () => {
      try {
        const res = await storageAdapter.list("customer:");
        if (res && res.keys && res.keys.length > 0) {
          const list = [];
          for (const key of res.keys) {
            try {
              const r = await storageAdapter.get(key);
              if (r) {
                const parsed = parseStoredValue(r.value);
                if (parsed) list.push(parsed);
              }
            } catch (e) {
              /* skip */
            }
          }
          list.sort((a, b) => a.name.localeCompare(b.name, "ja"));
          setCustomers(list);
        }
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  // 選択中の顧客の履歴をロード
  useEffect(() => {
    if (!currentCustomerId) {
      setHistory([]);
      return;
    }
    (async () => {
      try {
        const r = await storageAdapter.get(`history:${currentCustomerId}`);
        const list = r ? parseStoredValue(r.value) || [] : [];
        setHistory(Array.isArray(list) ? list : []);
      } catch (e) {
        setHistory([]);
      }
    })();
  }, [currentCustomerId]);

  const handleCreateCustomer = async (customer) => {
    try {
      const result = await storageAdapter.set(`customer:${customer.id}`, customer);
      if (!result) throw new Error("save returned falsy");
      setCustomers((prev) => [...prev, customer].sort((a, b) => a.name.localeCompare(b.name, "ja")));
      setCurrentCustomerId(customer.id);
      setLoadError(null);
    } catch (e) {
      console.error("customer save error", e);
      setLoadError("お客様情報の保存に失敗しました。（" + (e && e.message ? e.message : String(e)) + "）");
    }
  };

  const runDetection = async (canvas) => {
    if (!detectorRef.current) return null;
    const poses = await detectorRef.current.estimatePoses(canvas, { flipHorizontal: false });
    if (!poses || poses.length === 0) return { ok: false, missing: ["person_not_detected"] };
    return computeDistortion(poses[0].keypoints);
  };

  const handleCaptureFor = (which) => async (canvas) => {
    setBusy(true);
    try {
      const result = await runDetection(canvas);
      const img = new Image();
      img.src = canvas.toDataURL("image/jpeg", 0.85);
      await new Promise((resolve) => {
        img.onload = resolve;
      });
      const shot = { imageEl: img, result, width: canvas.width, height: canvas.height, takenAt: Date.now() };
      if (which === "before") {
        setBeforeShot(shot);
        setPhase("after");
      } else {
        setAfterShot(shot);
        setPhase("summary");
      }
    } catch (e) {
      console.error(e);
      setLoadError("姿勢の検出に失敗しました。もう一度撮影してください。");
    } finally {
      setBusy(false);
    }
  };

  const saveSession = async () => {
    if (!currentCustomerId || !beforeShot || !afterShot) return;
    setSaving(true);
    try {
      const entry = {
        id: "s_" + Date.now(),
        date: Date.now(),
        beforeScore: beforeShot.result.ok ? beforeShot.result.total : null,
        afterScore: afterShot.result.ok ? afterShot.result.total : null,
        beforeDetails: beforeShot.result.ok ? beforeShot.result.details : null,
        afterDetails: afterShot.result.ok ? afterShot.result.details : null,
      };
      const newHistory = [entry, ...history].slice(0, 50);
      const result = await storageAdapter.set(`history:${currentCustomerId}`, newHistory);
      if (!result) throw new Error("save failed");
      setHistory(newHistory);
    } catch (e) {
      console.error("session save error", e);
      setLoadError("記録の保存に失敗しました。（" + (e && e.message ? e.message : String(e)) + "）");
    } finally {
      setSaving(false);
    }
  };

  const resetFlow = () => {
    setBeforeShot(null);
    setAfterShot(null);
    setPhase("before");
    setLoadError(null);
  };

  const canProceed = modelStatus === "ready" && !busy;

  return (
    <div className="app-root">
      <style>{styles}</style>

      <header className="app-header">
        <div className="brand-mark">THIDA</div>
        <div className="brand-sub">姿勢歪みチェック</div>
      </header>

      <main className="app-main">
        <section className="card">
          <CustomerPicker
            customers={customers}
            currentId={currentCustomerId}
            onSelect={setCurrentCustomerId}
            onCreate={handleCreateCustomer}
          />
        </section>

        {modelStatus !== "ready" && (
          <section className="card status-card">
            {modelStatus === "loading" && (
              <p>骨格検出モデルを読み込んでいます...{modelProgress && <><br /><span style={{ opacity: 0.6, fontSize: "11px" }}>{modelProgress}</span></>}</p>
            )}
            {modelStatus === "error" && (
              <p className="error-text">
                モデルの読み込みに失敗しました。通信環境をご確認の上、再読み込みしてください。
                {modelErrorDetail && (
                  <><br /><span style={{ opacity: 0.7, fontSize: "11px" }}>詳細: {modelErrorDetail}</span></>
                )}
              </p>
            )}
            {modelStatus === "idle" && <p>準備中...</p>}
          </section>
        )}

        {loadError && (
          <section className="card status-card">
            <p className="error-text">{loadError}</p>
          </section>
        )}

        {!currentCustomerId && (
          <section className="card status-card">
            <p>お客様を選択、または新規追加してから撮影を開始してください。</p>
          </section>
        )}

        {currentCustomerId && phase === "before" && (
          <section className="card">
            <h2 className="section-title">STEP 1｜施術前の撮影</h2>
            <CameraCapture onCaptured={handleCaptureFor("before")} disabled={!canProceed} />
          </section>
        )}

        {currentCustomerId && beforeShot && phase !== "before" && (
          <section className="card">
            <h2 className="section-title">施術前の結果</h2>
            <div className="shot-result-row">
              <ResultOverlayCanvas
                imageEl={beforeShot.imageEl}
                result={beforeShot.result}
                width={beforeShot.width}
                height={beforeShot.height}
              />
              {beforeShot.result.ok ? (
                <div className="shot-result-side">
                  <ScoreGauge score={beforeShot.result.total} />
                  <ResultDetails result={beforeShot.result} />
                </div>
              ) : (
                <p className="error-text">
                  骨格を検出できませんでした。全身が映るように撮り直してください。
                </p>
              )}
            </div>
          </section>
        )}

        {currentCustomerId && phase === "after" && (
          <section className="card">
            <h2 className="section-title">STEP 2｜施術後の撮影</h2>
            <CameraCapture onCaptured={handleCaptureFor("after")} disabled={!canProceed} />
          </section>
        )}

        {currentCustomerId && phase === "summary" && afterShot && (
          <>
            <section className="card">
              <h2 className="section-title">施術後の結果</h2>
              <div className="shot-result-row">
                <ResultOverlayCanvas
                  imageEl={afterShot.imageEl}
                  result={afterShot.result}
                  width={afterShot.width}
                  height={afterShot.height}
                />
                {afterShot.result.ok ? (
                  <div className="shot-result-side">
                    <ScoreGauge score={afterShot.result.total} />
                    <ResultDetails result={afterShot.result} />
                  </div>
                ) : (
                  <p className="error-text">
                    骨格を検出できませんでした。全身が映るように撮り直してください。
                  </p>
                )}
              </div>
            </section>

            {beforeShot.result.ok && afterShot.result.ok && (
              <section className="card highlight-card">
                <h2 className="section-title">比較結果</h2>
                <div className="compare-row">
                  <div className="compare-item">
                    <span className="compare-label">施術前</span>
                    <span className="compare-score">{beforeShot.result.total}</span>
                  </div>
                  <div className="compare-arrow">→</div>
                  <div className="compare-item">
                    <span className="compare-label">施術後</span>
                    <span className="compare-score">{afterShot.result.total}</span>
                  </div>
                  <div className="compare-delta">
                    {afterShot.result.total - beforeShot.result.total <= 0 ? "▼" : "▲"}
                    {Math.abs(afterShot.result.total - beforeShot.result.total)}
                  </div>
                </div>

                <div className="reco-block">
                  <h3 className="reco-title">おすすめのトレーニング・ストレッチ</h3>
                  <ul className="reco-list">
                    {recommendTraining(afterShot.result).map((r, i) => (
                      <li key={i} className="reco-item">
                        <strong>{r.title}</strong>
                        <p>{r.detail}</p>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="reco-block next-visit-block">
                  <h3 className="reco-title">次回の整体タイミング</h3>
                  {(() => {
                    const rec = recommendNextVisit(afterShot.result.total);
                    return (
                      <div className="next-visit">
                        <div className="next-visit-days">{rec.days}</div>
                        <p>{rec.note}</p>
                      </div>
                    );
                  })()}
                </div>

                <div className="action-row">
                  <button className="btn btn-primary" onClick={saveSession} disabled={saving}>
                    {saving ? "保存中..." : "この結果を記録する"}
                  </button>
                  <button className="btn btn-ghost" onClick={resetFlow}>
                    新しく撮影する
                  </button>
                </div>
              </section>
            )}
          </>
        )}

        {currentCustomerId && history.length > 0 && (
          <section className="card">
            <h2 className="section-title">これまでの履歴</h2>
            <div className="history-list">
              {history.map((h) => (
                <div key={h.id} className="history-item">
                  <div className="history-date">
                    {new Date(h.date).toLocaleDateString("ja-JP", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </div>
                  <div className="history-scores">
                    <span>施術前: {h.beforeScore ?? "-"}</span>
                    <span className="history-sep">→</span>
                    <span>施術後: {h.afterScore ?? "-"}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

const styles = `
:root {
  --bg: #12181f;
  --bg-panel: #1a2029;
  --line: #2a3340;
  --ink: #f4efe6;
  --ink-dim: #9aa7b5;
  --coral: #e2603f;
  --coral-deep: #c94f31;
  --azul: #3a7fe0;
  --sun: #f2c14e;
}
* { box-sizing: border-box; }
.app-root {
  min-height: 100vh;
  background: radial-gradient(circle at 20% -10%, #1c2530 0%, var(--bg) 55%);
  color: var(--ink);
  font-family: "Hiragino Sans", "Noto Sans JP", -apple-system, sans-serif;
  padding-bottom: 48px;
}
.app-header {
  padding: 28px 20px 18px;
  display: flex;
  align-items: baseline;
  gap: 12px;
  border-bottom: 1px solid var(--line);
}
.brand-mark {
  font-size: 26px;
  font-weight: 900;
  letter-spacing: 0.08em;
  color: var(--sun);
}
.brand-sub {
  font-size: 14px;
  color: var(--ink-dim);
  letter-spacing: 0.04em;
}
.app-main {
  max-width: 640px;
  margin: 0 auto;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.card {
  background: var(--bg-panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px;
}
.status-card { text-align: center; color: var(--ink-dim); }
.section-title {
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.04em;
  margin: 0 0 14px;
  color: var(--sun);
}
.field-label {
  display: block;
  font-size: 13px;
  color: var(--ink-dim);
  margin-bottom: 6px;
}
.select-input, .text-input {
  width: 100%;
  background: #0f1520;
  border: 1px solid var(--line);
  color: var(--ink);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 15px;
}
.customer-new {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
.customer-new .text-input { flex: 1; }
.btn {
  border: none;
  border-radius: 999px;
  padding: 12px 20px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.15s ease, opacity 0.15s ease;
}
.btn:active { transform: scale(0.97); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--coral); color: #fff; }
.btn-secondary { background: var(--azul); color: #fff; }
.btn-ghost { background: transparent; color: var(--ink-dim); border: 1px solid var(--line); }
.camera-box { display: flex; flex-direction: column; gap: 12px; }
.camera-frame {
  position: relative;
  width: 100%;
  aspect-ratio: 3 / 4;
  background: #05070a;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--line);
}
.camera-video { width: 100%; height: 100%; object-fit: cover; }
.camera-placeholder {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--ink-dim); font-size: 13px; text-align: center; padding: 20px;
}
.camera-controls { display: flex; gap: 10px; flex-wrap: wrap; }
.hint-text { font-size: 12px; color: var(--ink-dim); margin: 0; }
.error-text { color: #ff8a7a; font-size: 13px; }
.shot-result-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.overlay-canvas {
  width: 200px;
  max-width: 100%;
  border-radius: 10px;
  border: 1px solid var(--line);
}
.shot-result-side {
  flex: 1;
  min-width: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}
.gauge-wrap { display: flex; flex-direction: column; align-items: center; }
.gauge-label { font-size: 13px; font-weight: 800; margin-top: 4px; }
.detail-grid {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.detail-item {
  background: #0f1520;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.detail-name { font-size: 11px; color: var(--ink-dim); }
.detail-value { font-size: 15px; font-weight: 700; }
.highlight-card { border-color: var(--coral-deep); }
.compare-row {
  display: flex;
  align-items: center;
  gap: 14px;
  justify-content: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.compare-item { text-align: center; }
.compare-label { display: block; font-size: 12px; color: var(--ink-dim); }
.compare-score { font-size: 28px; font-weight: 900; }
.compare-arrow { font-size: 20px; color: var(--ink-dim); }
.compare-delta {
  font-size: 16px;
  font-weight: 800;
  color: var(--sun);
  background: #0f1520;
  border-radius: 999px;
  padding: 6px 14px;
}
.reco-block { margin-bottom: 18px; }
.reco-title {
  font-size: 14px;
  font-weight: 800;
  color: var(--ink);
  margin: 0 0 10px;
  border-left: 3px solid var(--coral);
  padding-left: 8px;
}
.reco-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.reco-item {
  background: #0f1520;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 14px;
}
.reco-item strong { display: block; margin-bottom: 4px; color: var(--sun); font-size: 13px; }
.reco-item p { margin: 0; font-size: 13px; color: var(--ink-dim); line-height: 1.6; }
.next-visit {
  background: #0f1520;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px;
  text-align: center;
}
.next-visit-days { font-size: 22px; font-weight: 900; color: var(--azul); margin-bottom: 6px; }
.next-visit p { margin: 0; font-size: 13px; color: var(--ink-dim); }
.action-row { display: flex; gap: 10px; flex-wrap: wrap; }
.history-list { display: flex; flex-direction: column; gap: 8px; }
.history-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #0f1520;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 13px;
  flex-wrap: wrap;
  gap: 6px;
}
.history-date { color: var(--ink-dim); }
.history-scores { display: flex; gap: 6px; align-items: center; }
.history-sep { color: var(--ink-dim); }
@media (max-width: 420px) {
  .overlay-canvas { width: 100%; }
}
`;
