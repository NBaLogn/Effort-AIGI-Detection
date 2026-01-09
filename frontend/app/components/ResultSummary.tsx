"use client";

import styles from "./ResultSummary.module.css";

interface ResultSummaryProps {
  total: number;
  realCount: number;
  fakeCount: number;
  realPercent: number;
  fakePercent: number;
}

const formatPercent = (value: number) => value.toFixed(1);

export default function ResultSummary({
  total,
  realCount,
  fakeCount,
  realPercent,
  fakePercent,
}: ResultSummaryProps) {
  return (
    <section className={styles.summary} aria-live="polite">
      <div className={styles.heading}>
        <div className={styles.headingText}>
          <p className={styles.label}>Latest batch</p>
          <h2 className={styles.title}>Upload summary</h2>
        </div>
        <span className={styles.totalBadge}>{total} images</span>
      </div>

      <div className={styles.stats}>
        <article className={styles.statCard}>
          <div className={styles.statHeader}>
            <span className={styles.statLabel}>Real</span>
            <span className={styles.statPercent}>{formatPercent(realPercent)}%</span>
          </div>
          <p className={styles.statValue}>{realCount}</p>
          <p className={styles.statCaption}>Labeled real</p>
        </article>

        <article className={styles.statCard}>
          <div className={styles.statHeader}>
            <span className={`${styles.statLabel} ${styles.fakeLabel}`}>Fake</span>
            <span className={styles.statPercent}>{formatPercent(fakePercent)}%</span>
          </div>
          <p className={styles.statValue}>{fakeCount}</p>
          <p className={styles.statCaption}>Labeled fake</p>
        </article>
      </div>
    </section>
  );
}
