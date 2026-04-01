import type { GeneratedVariant } from '../types/variant';

interface PoseNavigatorProps {
  variants: GeneratedVariant[];
  activeVariantId: number | null;
  onSelectVariant: (variant: GeneratedVariant) => void;
  itemLabel?: string;
}

export function PoseNavigator({ variants, activeVariantId, onSelectVariant, itemLabel = 'Pose' }: PoseNavigatorProps) {
  if (variants.length === 0) return null;

  const activeIndex = variants.findIndex(v => v.id === activeVariantId);
  const current = activeIndex >= 0 ? activeIndex + 1 : 0;
  const total = variants.length;

  const goNext = () => {
    const nextIdx = (activeIndex + 1) % variants.length;
    onSelectVariant(variants[nextIdx]);
  };

  const goPrev = () => {
    const prevIdx = (activeIndex - 1 + variants.length) % variants.length;
    onSelectVariant(variants[prevIdx]);
  };

  return (
    <div className="lp-pose-nav">
      <button className="lp-pose-nav__btn" onClick={goPrev} aria-label={`Previous ${itemLabel.toLowerCase()}`}>&lsaquo;</button>
      <span className="lp-pose-nav__label">{itemLabel} {current}/{total}</span>
      <button className="lp-pose-nav__btn" onClick={goNext} aria-label={`Next ${itemLabel.toLowerCase()}`}>&rsaquo;</button>
    </div>
  );
}
