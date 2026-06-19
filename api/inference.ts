import * as ort from "onnxruntime-node";

const LABELS = [
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate"
];

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { input_ids, attention_mask } = req.body;

  try {
    const session = await ort.InferenceSession.create(
      "./public/model.onnx"
    );

    const results = await session.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(input_ids), [1, input_ids.length]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(attention_mask), [1, attention_mask.length])
    });

    const logits = results.logits.data as number[];

    const output: any = {};
    LABELS.forEach((label, i) => {
      output[label] = sigmoid(logits[i]);
    });

    return res.status(200).json(output);

  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
}