"use client";

import { useState, useEffect, useCallback, useRef } from "react";

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 3000,
  deps: any[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);

  // Always keep the ref current so the interval uses the latest fetcher
  fetcherRef.current = fetcher;

  const fetchData = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      if (mountedRef.current) {
        setData(result);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err as Error);
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  const refetch = useCallback(() => {
    setLoading(true);
    fetchData();
  }, [fetchData]);

  // Re-fetch immediately when deps change, then poll on interval
  useEffect(() => {
    mountedRef.current = true;
    setLoading(true);
    fetchData();
    const id = setInterval(fetchData, intervalMs);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, fetchData, ...deps]);

  return { data, error, loading, refetch };
}
