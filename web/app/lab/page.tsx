import { getModelComparison, getAccuracy } from "@/lib/api";
import { ModelLab } from "@/components/model-lab/lab";

export const dynamic = "force-dynamic";

export default async function LabPage() {
  let comparison;
  let accuracy;
  try {
    [comparison, accuracy] = await Promise.all([
      getModelComparison(),
      getAccuracy(),
    ]);
  } catch {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <p className="font-mono text-sm text-[var(--red)]">
          Could not load model comparison — is the API running?
        </p>
      </div>
    );
  }

  return <ModelLab comparison={comparison} accuracy={accuracy} />;
}
