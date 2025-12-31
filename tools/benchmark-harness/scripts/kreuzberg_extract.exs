#!/usr/bin/env elixir
# Kreuzberg Elixir extraction wrapper for benchmark harness.
#
# Supports two modes:
# - sync: extract_file/2 - synchronous extraction
# - batch: batch_extract_files/2 - synchronous batch extraction

require Logger

# Configure Logger to suppress debug messages and use stderr for all output
# This ensures only the JSON result goes to stdout
Logger.configure(level: :warning)
Logger.configure_backend(:console, device: :standard_error)

debug = System.get_env("KREUZBERG_BENCHMARK_DEBUG", "false") == "true"

defmodule KreuzbergExtract do
  @moduledoc """
  Kreuzberg extraction wrapper for benchmarking.
  """

  def debug_log(message) when is_binary(message) do
    if System.get_env("KREUZBERG_BENCHMARK_DEBUG", "false") == "true" do
      IO.write(:stderr, "[DEBUG] #{DateTime.utc_now() |> DateTime.to_iso8601()} - #{message}\n")
    end
  end

  def debug_log(_), do: nil

  @doc """
  Convert a struct to a plain map for JSON encoding.
  Handles nested structs and removes nil values.
  """
  def struct_to_map(nil), do: %{}
  def struct_to_map(struct) when is_struct(struct) do
    struct
    |> Map.from_struct()
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Enum.map(fn {k, v} -> {Atom.to_string(k), struct_to_map(v)} end)
    |> Map.new()
  end
  def struct_to_map(value), do: value

  @doc """
  Extract a single file synchronously.
  """
  def extract_sync(file_path) do
    debug_log("=== SYNC EXTRACTION START ===")
    debug_log("Input: file_path=#{file_path}")
    debug_log("File exists: #{File.exists?(file_path)}")

    if File.exists?(file_path) do
      size = File.stat!(file_path).size
      debug_log("File size: #{size} bytes")
    end

    start_time = System.monotonic_time(:millisecond)
    start_wall = DateTime.utc_now()
    debug_log("Timing start (monotonic): #{start_time}, wall: #{DateTime.to_iso8601(start_wall)}")

    result = Kreuzberg.extract_file(file_path)

    end_time = System.monotonic_time(:millisecond)
    end_wall = DateTime.utc_now()
    duration_ms = end_time - start_time

    debug_log("Timing end (monotonic): #{end_time}, wall: #{DateTime.to_iso8601(end_wall)}")
    debug_log("Duration (milliseconds): #{duration_ms}")

    case result do
      {:ok, extraction_result} ->
        debug_log("Result class: Kreuzberg.ExtractionResult")
        debug_log("Result has content: true")
        debug_log("Content length: #{String.length(extraction_result.content)} characters")
        debug_log("Result has metadata: true")
        debug_log("Metadata type: map")

        payload = %{
          "content" => extraction_result.content,
          "metadata" => struct_to_map(extraction_result.metadata),
          "_extraction_time_ms" => duration_ms
        }

        json_size = payload |> Jason.encode!() |> byte_size()
        debug_log("Output JSON size: #{json_size} bytes")
        debug_log("=== SYNC EXTRACTION END ===")
        {:ok, payload}

      {:error, reason} ->
        debug_log("ERROR during sync extraction: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Extract multiple files in batch mode.
  """
  def extract_batch(file_paths) do
    debug_log("=== BATCH EXTRACTION START ===")
    debug_log("Input: #{length(file_paths)} files")

    file_paths
    |> Enum.with_index()
    |> Enum.each(fn {path, idx} ->
      exists = File.exists?(path)
      size = if exists, do: File.stat!(path).size, else: "N/A"
      debug_log("  [#{idx}] #{path} (exists: #{exists}, size: #{size} bytes)")
    end)

    start_time = System.monotonic_time(:millisecond)
    start_wall = DateTime.utc_now()
    debug_log("Timing start (monotonic): #{start_time}, wall: #{DateTime.to_iso8601(start_wall)}")

    result = Kreuzberg.batch_extract_files(file_paths)

    end_time = System.monotonic_time(:millisecond)
    end_wall = DateTime.utc_now()
    total_duration_ms = end_time - start_time

    debug_log("Timing end (monotonic): #{end_time}, wall: #{DateTime.to_iso8601(end_wall)}")
    debug_log("Total duration (milliseconds): #{total_duration_ms}")

    case result do
      {:ok, results} ->
        debug_log("Results count: #{length(results)}")

        per_file_duration_ms =
          if length(file_paths) > 0 do
            total_duration_ms / length(file_paths)
          else
            0
          end

        debug_log("Per-file average duration (milliseconds): #{per_file_duration_ms}")

        results_with_timing =
          results
          |> Enum.with_index()
          |> Enum.map(fn {extraction_result, idx} ->
            content_length = String.length(extraction_result.content || "")
            debug_log("  Result[#{idx}] - content length: #{content_length}, has metadata: true")

            %{
              "content" => extraction_result.content,
              "metadata" => struct_to_map(extraction_result.metadata),
              "_extraction_time_ms" => per_file_duration_ms,
              "_batch_total_ms" => total_duration_ms
            }
          end)

        debug_log("=== BATCH EXTRACTION END ===")
        {:ok, results_with_timing}

      {:error, reason} ->
        debug_log("ERROR during batch extraction: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Main entry point for the script.
  """
  def main(args) do
    debug_log("Elixir script started")
    debug_log("ARGV: #{inspect(args)}")
    debug_log("ARGV length: #{length(args)}")

    case args do
      [] ->
        IO.puts(:stderr, "Usage: kreuzberg_extract.exs <mode> <file_path> [additional_files...]")
        IO.puts(:stderr, "Modes: sync, batch")
        IO.puts(:stderr, "Debug mode: set KREUZBERG_BENCHMARK_DEBUG=true to enable debug logging to stderr")
        System.halt(1)

      [mode | file_paths] ->
        debug_log("Mode: #{mode}")
        debug_log("File paths (#{length(file_paths)}): #{inspect(file_paths)}")

        case mode do
          "sync" ->
            if length(file_paths) != 1 do
              IO.puts(:stderr, "Error: sync mode requires exactly one file")
              System.halt(1)
            end

            debug_log("Executing sync mode with file: #{hd(file_paths)}")

            case extract_sync(hd(file_paths)) do
              {:ok, payload} ->
                json = Jason.encode!(payload)
                debug_log("Output JSON: #{json}")
                IO.write(json)

              {:error, reason} ->
                IO.puts(:stderr, "Error extracting with Kreuzberg: #{inspect(reason)}")
                System.halt(1)
            end

          "batch" ->
            if length(file_paths) == 0 do
              IO.puts(:stderr, "Error: batch mode requires at least one file")
              System.halt(1)
            end

            debug_log("Executing batch mode with #{length(file_paths)} files")

            case extract_batch(file_paths) do
              {:ok, results} ->
                json =
                  if length(file_paths) == 1 do
                    Jason.encode!(hd(results))
                  else
                    Jason.encode!(results)
                  end

                debug_log("Output JSON: #{String.slice(json, 0..200)}...")
                IO.write(json)

              {:error, reason} ->
                IO.puts(:stderr, "Error extracting with Kreuzberg: #{inspect(reason)}")
                System.halt(1)
            end

          _ ->
            IO.puts(:stderr, "Error: Unknown mode '#{mode}'. Use sync or batch")
            System.halt(1)
        end

        debug_log("Script completed successfully")
    end
  rescue
    e ->
      debug_log("FATAL ERROR: #{inspect(e)}")
      debug_log("Backtrace: #{inspect(__STACKTRACE__)}")
      IO.puts(:stderr, "Error extracting with Kreuzberg: #{inspect(e)}")
      System.halt(1)
  end
end

# Start the application and run main
{:ok, _apps} = Application.ensure_all_started(:kreuzberg)

# Parse args and run
args = System.argv()
KreuzbergExtract.main(args)
