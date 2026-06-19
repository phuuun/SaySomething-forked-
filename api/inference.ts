// api/inference.ts
// This replaces your Gemini version with real Hugging Face Inference API
// Place this file in: /api/inference.ts (Vercel auto-detects it)

export default async function handler(req: any, res: any) {
  // Only POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { text, model } = req.body;

  // Validate input
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'text is required and must be a string' });
  }

  // Only DistilBERT is available via Hugging Face free tier
  if (model !== 'DistilBERT' && model !== 'distilbert') {
    return res.status(400).json({
      error: `Model '${model}' not available. Only 'DistilBERT' is deployed. (No more Gemini!)`
    });
  }

  try {
    // Get credentials from environment variables
    const HF_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
    const HF_MODEL_ID = process.env.HUGGINGFACE_MODEL_ID;

    if (!HF_API_TOKEN || !HF_MODEL_ID) {
      console.error('Missing HF environment variables');
      return res.status(500).json({
        error: 'Server not configured. Ask the admin to set HUGGINGFACE_API_TOKEN and HUGGINGFACE_MODEL_ID.'
      });
    }

    console.log(`[Inference] Processing: "${text.substring(0, 50)}..."  via ${HF_MODEL_ID}`);

    // Call Hugging Face Inference API
    const hfResponse = await fetch(
      `https://api-inference.huggingface.co/models/${HF_MODEL_ID}`,
      {
        headers: {
          Authorization: `Bearer ${HF_API_TOKEN}`,
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify({
          inputs: text,
          options: {
            wait_for_model: true  // Wait if model is loading
          }
        })
      }
    );

    if (!hfResponse.ok) {
      const errorData = await hfResponse.json().catch(() => ({}));
      console.error('HF API error:', hfResponse.status, errorData);

      // Helpful error messages
      if (hfResponse.status === 503) {
        return res.status(503).json({
          error: 'Model is loading. Try again in 30 seconds.'
        });
      }

      return res.status(hfResponse.status).json({
        error: errorData.error || `Inference failed (${hfResponse.status})`
      });
    }

    const result = await hfResponse.json();

    // Convert Hugging Face output to our format
    const scores = parseHFResponse(result);

    console.log(`[Inference] Result:`, scores);
    return res.status(200).json(scores);

  } catch (error: any) {
    console.error('Inference error:', error);
    return res.status(500).json({
      error: error.message || 'Internal server error'
    });
  }
}

/**
 * Parse Hugging Face Inference API response
 * HF returns logits like: [[logit1, logit2, logit3, logit4, logit5, logit6]]
 * Convert to probabilities using sigmoid
 */
function parseHFResponse(result: any): Record<string, number> {
  const LABELS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
  ];

  // Sigmoid function: converts logits to probabilities (0-1)
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

  // Extract logits from HF response
  // Format is usually: [[logit1, logit2, ...]] or sometimes just [logit1, logit2, ...]
  let logits: number[];

  if (Array.isArray(result)) {
    if (Array.isArray(result[0])) {
      logits = result[0];  // [[...]] format
    } else {
      logits = result;     // [...] format
    }
  } else if (result.logits) {
    logits = result.logits;
  } else if (result[0]?.logits) {
    logits = result[0].logits;
  } else {
    // Fallback: assume result is directly the logits
    logits = Object.values(result) as number[];
  }

  // Convert logits to probabilities
  const scores: Record<string, number> = {};
  LABELS.forEach((label, i) => {
    // Clamp logits to reasonable range to avoid overflow
    const logit = Math.max(-10, Math.min(10, logits[i] || 0));
    const prob = sigmoid(logit);
    scores[label] = Math.round(prob * 1000) / 1000;  // Round to 3 decimals
  });

  return scores;
}
