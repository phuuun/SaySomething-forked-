export async function runInference(text: string, model: string) {
  const response = await fetch('/api/inference', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, model }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const msg = errorData.error || `Inference failed (${response.status})`;

    // Give a helpful error if the Flask server isn't running
    if (response.status === 503) {
      throw new Error(`Model not ready: ${msg}`);
    }
    if (response.status === 0 || msg.includes('Failed to fetch')) {
      throw new Error(
        'Cannot reach model server. Make sure server_models.py is running:\n  python server_models.py'
      );
    }
    throw new Error(msg);
  }

  return response.json();
}
