import { useEffect, useState, type CSSProperties } from 'react';
import type { GeneratedVariant } from '../types/variant';
import { collectDetailMetrics, formatMetricValue, metricDisplayLabel, metricTone } from '../utils/variant-metrics';

export interface ResultsWorkspaceProps {
  variants: GeneratedVariant[];
  activeVariantId: number | null;
  activeVariant: GeneratedVariant | null;
  onSelectVariant: (variant: GeneratedVariant) => void;
  isGenerating: boolean;
  generateProgress: number;
  statusMessage: string;
  hasError: boolean;
  itemLabel?: string;
  open?: boolean;
  onToggleOpen?: (open: boolean) => void;
  focusMode?: boolean;
  className?: string;
  style?: CSSProperties;
}

const metric = (value: number | null, formatter: (numeric: number) => string): string => (
  typeof value === 'number' && Number.isFinite(value) ? formatter(value) : 'n/a'
);

const joinClasses = (...tokens: Array<string | false | null | undefined>): string => tokens.filter(Boolean).join(' ');

export function ResultsWorkspace({
  variants,
  activeVariantId,
  activeVariant,
  onSelectVariant,
  isGenerating,
  generateProgress,
  statusMessage,
  hasError,
  itemLabel = 'pose',
  open,
  onToggleOpen,
  focusMode = false,
  className = '',
  style,
}: ResultsWorkspaceProps) {
  const hasVariants = variants.length > 0;
  const [isOpenInternal, setIsOpenInternal] = useState(false);
  const shouldReveal = hasVariants || isGenerating || hasError;
  const isControlled = typeof open === 'boolean';
  const isOpen = isControlled ? open : isOpenInternal;
  const detailMetrics = activeVariant ? collectDetailMetrics(activeVariant).slice(0, 4) : [];
  const drawerClassName = joinClasses(
    'lp-results-drawer',
    isOpen ? 'lp-results-drawer--open' : 'lp-results-drawer--collapsed',
    focusMode && 'lp-results-drawer--focus',
    className,
  );

  const setOpen = (next: boolean) => {
    if (isControlled) {
      onToggleOpen?.(next);
      return;
    }
    setIsOpenInternal(next);
  };

  useEffect(() => {
    if (!isControlled && shouldReveal) {
      setOpen(true);
    }
  }, [shouldReveal]);

  const progressValue = Math.max(0, Math.min(100, Math.round(generateProgress * 100)));
  const loadingBarWidth = `${Math.max(8, progressValue)}%`;
  const countLabel = hasVariants ? `${variants.length} ${itemLabel}${variants.length === 1 ? '' : 's'}` : 'Empty';

  return (
    <section
      className={drawerClassName}
      data-state={isOpen ? 'open' : 'closed'}
      style={style}
    >
      <div className="lp-results-drawer__header">
        <div className="lp-results-drawer__summary">
          <span className="lp-results-drawer__eyebrow">Results</span>
          <strong className="lp-results-drawer__title">{countLabel}</strong>
          {(isGenerating || hasError) && <span className="lp-results-drawer__status">{statusMessage}</span>}
        </div>

        <button
          className="lp-results-drawer__toggle lp-btn lp-btn--ghost"
          onClick={() => setOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-label={isOpen ? 'Collapse results drawer' : 'Expand results drawer'}
          title={isOpen ? 'Collapse results drawer' : 'Expand results drawer'}
        >
          {isOpen ? '▴' : '▾'}
        </button>
      </div>

      {isOpen && (
        <div className="lp-results-drawer__body">
          {isGenerating && (
            <div
              className="lp-results-drawer__loading"
              role="progressbar"
              aria-label="Generation progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progressValue}
              aria-valuetext={`${progressValue}%`}
            >
              <div className="lp-results-drawer__loading-bar" style={{ width: loadingBarWidth }} />
            </div>
          )}

          {!hasVariants ? (
            <div className="lp-results-drawer__empty">
              <p>{statusMessage}</p>
            </div>
          ) : (
            <div className="lp-results-drawer__content">
              <div className="lp-results-drawer__strip">
                {variants.map((variant) => (
                  <button
                    key={variant.id}
                    className={`lp-result-card ${activeVariantId === variant.id ? 'lp-result-card--active' : ''}`}
                    onClick={() => onSelectVariant(variant)}
                  >
                    <div className="lp-result-card__header">
                      <span>{variant.label}</span>
                      <span style={{ color: metricTone(variant.primaryMetricKey, variant.primaryMetricValue) }}>
                        {formatMetricValue(variant.primaryMetricKey, variant.primaryMetricValue)}
                      </span>
                    </div>
                    <div className="lp-result-card__metrics">
                      <span>{metric(variant.rank, (value) => `#${Math.round(value)}`)}</span>
                      <span>{metric(variant.rmsd, (value) => `${value.toFixed(2)}A`)}</span>
                    </div>
                  </button>
                ))}
              </div>

              <div className="lp-results-drawer__detail">
                {activeVariant ? (
                  <>
                    <div className="lp-results-drawer__detail-header">
                      <h3>{activeVariant.label}</h3>
                      <span
                        className="lp-results-drawer__detail-score"
                        style={{ color: metricTone(activeVariant.primaryMetricKey, activeVariant.primaryMetricValue) }}
                      >
                        {formatMetricValue(activeVariant.primaryMetricKey, activeVariant.primaryMetricValue)}
                      </span>
                    </div>
                    <dl className="lp-data-list">
                      <div className="lp-data-list__row">
                        <dt>{metricDisplayLabel(activeVariant.primaryMetricKey)}</dt>
                        <dd>{formatMetricValue(activeVariant.primaryMetricKey, activeVariant.primaryMetricValue)}</dd>
                      </div>
                      <div className="lp-data-list__row">
                        <dt>Rank</dt>
                        <dd>{metric(activeVariant.rank, (value) => `#${Math.round(value)}`)}</dd>
                      </div>
                      <div className="lp-data-list__row">
                        <dt>RMSD</dt>
                        <dd>{metric(activeVariant.rmsd, (value) => `${value.toFixed(2)}A`)}</dd>
                      </div>
                      <div className="lp-data-list__row">
                        <dt>Duplicates</dt>
                        <dd>{metric(activeVariant.duplicateCount, (value) => `${Math.round(value)}`)}</dd>
                      </div>
                      {detailMetrics.map(({ key, value }) => (
                        <div className="lp-data-list__row" key={key}>
                          <dt>{metricDisplayLabel(key)}</dt>
                          <dd>{formatMetricValue(key, value)}</dd>
                        </div>
                      ))}
                    </dl>
                    <p className="lp-card__mono">{activeVariant.smiles ?? 'No SMILES available'}</p>
                  </>
                ) : (
                  <div className="lp-results-drawer__empty">
                    <p>Select a pose to inspect it.</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
