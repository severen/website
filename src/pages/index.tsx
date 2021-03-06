import React from 'react';

import PostList from '../components/PostList';

interface Props {
  data: {
    allMarkdownRemark: any;
  };
}

const IndexPage = ({ data }: Props) => {
  return <PostList postEdges={data.allMarkdownRemark.edges} />;
};

export default IndexPage;

export const pageQuery = graphql`
  query IndexQuery {
    allMarkdownRemark(
      limit: 2000
      sort: { fields: [frontmatter___date], order: DESC }
    ) {
      edges {
        node {
          excerpt
          timeToRead
          frontmatter {
            title
            date(formatString: "DD/MM/YYYY")
            path
          }
        }
      }
    }
  }
`;
