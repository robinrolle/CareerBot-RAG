'use client';

import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import React, { useState } from 'react';
import FileUpload from './FileUpload';
import { Tabs, Tab } from "@nextui-org/react";
import './style/Tabs.css';

export default function Home() {
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  const handleFileUpload = (filename) => {
    setUploadedFilename(filename);
    setAnalyzed(false);
  };

  const handleFileRemove = () => {
    setUploadedFilename(null);
    setAnalyzed(false);
  };

  const handleSubmit = async () => {
    if (!uploadedFilename) return;

    setLoading(true);
    try {
      setTimeout(() => {
        const mockResponse = {
          skills: ['JavaScript', 'React', 'Node.js'],
          occupations: ['Frontend Developer', 'Full Stack Developer', 'Software Engineer']
        };
        setAnalyzed(true);
        setLoading(false);
      }, 2000);
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Head>
        {/* TODO add title and logo */}
      </Head>

      <header className="flex justify-between items-center p-4 bg-white shadow">
        <div>
          <Image src="/media/3amk_full.png" alt="Logo" width={100} height={75} />
        </div>
        <nav className="flex space-x-4">
          <Link href="/about" className="text-gray-700 hover:text-gray-900">
            About
          </Link>
          <Link href="/contact" className="text-gray-700 hover:text-gray-900">
            Contact
          </Link>
        </nav>
      </header>

      <main className="container mx-auto px-4 py-16">
        <h1 className="text-5xl font-bold text-center mb-4">CareerBot</h1>
        <p className="text-xl text-center mb-8">Analyze your resume and build your profile!</p>

        <div className="flex flex-col items-center">
          <div className="w-full max-w-md">
            <FileUpload onFileUpload={handleFileUpload} onFileRemove={handleFileRemove} />
          </div>

          {uploadedFilename && (
            <button
              className="bg-blue-500 text-white px-4 py-2 mt-4 rounded shadow hover:bg-blue-600 w-full max-w-md"
              onClick={handleSubmit}
              disabled={loading || analyzed}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          )}
        </div>

        {analyzed && (
          <div className="selectionContainer mt-16">
            <h2 className="text-4xl font-semibold">Selection</h2>
            <Tabs key="types" className="customTabs">
              <Tab key="skills" title="Skills" className="customTab">
                <div>skills container</div>
              </Tab>
              <Tab key="occupations" title="Occupations" className="customTab">
                <div>occupations container</div>
              </Tab>
            </Tabs>
          </div>
        )}
      </main>
    </div>
  );
}
